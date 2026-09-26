"""The :class:`Wrapper` -- intercept everything around any upstream provider.

A wrapper is the thing you design around, FastAPI-style:

    wrapper = Wrapper(providers={...}, transport=...)

    @wrapper.before("route")        # tag-keyed event (fires for that tag)
    def rewrite(ctx, tag):
        if tag.value:
            ctx.body["model"] = tag.value

    @wrapper.after                  # per-call response hook (every call)
    def shape(payload, ctx):
        return payload

Non-streaming calls go through ``before`` -> transport -> ``after``.
Streaming calls go through ``before`` -> transport -> per-chunk ``stream``.
Any failure passes through ``error`` hooks before becoming an OpenAI-shaped
JSON error.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .config import Config
from .core import DONE
from .core.context import RequestContext
from .core.errors import ApiError
from .core.events import SCRATCH_KEY, STATE_KEY, EventRegistry
from .core.hooks import HookKind, HookRegistry, maybe_await
from .core.protocol import PreparedRequest, Transport, UpstreamStream
from .core.router import ModelRouter
from .providers import build_providers
from .telemetry import CallRecord, Telemetry

__all__ = ["GatewayResponse", "StreamHandle", "Wrapper"]

logger = logging.getLogger("margAI.wrapper")

_SSE_DONE = "data: [DONE]\n\n"


@dataclass
class GatewayResponse:
    body: dict[str, Any]
    status: int = 200


class Wrapper:
    def __init__(
        self,
        *,
        providers: dict[str, Any],
        transport: Transport,
        prefix: str = "margAI",
        expose: str = "prefixed",
        default_provider: str | None = None,
        telemetry: Telemetry | None = None,
        config: Config | None = None,
        name: str = "margAI",
    ) -> None:
        self.name = name
        self.providers = providers
        self.transport = transport
        self.config = config
        self._hooks = HookRegistry()
        self._events = EventRegistry()
        self._events_wired = False
        self.router = ModelRouter(
            providers, prefix=prefix, expose=expose, default_provider=default_provider
        )
        self.telemetry = telemetry or Telemetry(_disabled_telemetry_config())
        self._catalog_cache: list[dict[str, Any]] | None = None
        self._catalog_at = 0.0
        self._catalog_ttl = config.gateway.models_cache_ttl if config else 300.0

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        transport: Transport | None = None,
        *,
        load_hooks: bool = True,
        name: str = "margAI",
    ) -> "Wrapper":
        from .transport.httpx import HttpxTransport

        transport = transport or HttpxTransport(timeout=config.gateway.timeout)
        wrapper = cls(
            providers=build_providers(config),
            transport=transport,
            prefix=config.gateway.prefix,
            expose=config.gateway.expose,
            default_provider=config.gateway.default_provider,
            telemetry=Telemetry(config.telemetry),
            config=config,
            name=name,
        )
        if load_hooks:
            for entry in config.hooks.load:
                _load_hook_module(entry, wrapper)
        return wrapper

    async def aclose(self) -> None:
        closer = getattr(self.transport, "aclose", None)
        if closer is not None:
            await closer()

    # -- event decorators ----------------------------------------------------
    #
    # FastAPI-style surface. Each decorator has two modes selected by whether
    # the first positional arg is a string (or `tag=` is given):
    #
    #   @app.before("route")            # tag-keyed event: fn(ctx, tag)
    #   @app.before                     # per-call hook: fn(ctx)
    #
    # Tag-keyed events are stored on the wrapper's EventRegistry (see
    # `app.events`) and dispatched automatically at the matching phase for
    # every call whose `ctx.state[STATE_KEY]` includes the tag. `name=`,
    # when given without a tag, is the per-call hook's display name.

    def before(self, fn=None, *, tag=None, name=None, order=0):
        """Request phase.

        - ``@app.before("name")`` / ``@app.before(tag="name")``: an event
          handler ``fn(ctx, tag)`` run when the call's tags include ``name``.
        - ``@app.before``: a per-call hook ``fn(ctx)`` on every request.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            self._ensure_event_dispatch()
            return self._events.before(tag)
        return self._decorate(HookKind.REQUEST, fn, name, order)

    def after(self, fn=None, *, tag=None, name=None, order=0):
        """Response phase (non-streaming).

        - ``@app.after("name")``: event handler ``fn(payload, ctx, tag) ->
          payload``.
        - ``@app.after``: per-call hook ``fn(payload, ctx) -> payload``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            self._ensure_event_dispatch()
            return self._events.after(tag)
        return self._decorate(HookKind.RESPONSE, fn, name, order)

    def stream(self, fn=None, *, tag=None, name=None, order=0):
        """Stream phase (per SSE chunk).

        - ``@app.stream("name")``: event handler ``fn(chunk, ctx, tag,
          scratch) -> chunk | None`` (``None`` drops the chunk).
        - ``@app.stream``: per-call hook ``fn(chunk, ctx) -> chunk | None``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            self._ensure_event_dispatch()
            return self._events.stream(tag)
        return self._decorate(HookKind.STREAM, fn, name, order)

    def error(self, fn=None, *, tag=None, name=None, order=0):
        """Failure phase.

        - ``@app.error("name")``: event handler ``fn(exc, ctx, tag) ->
          dict | None`` (a dict takes over the error body).
        - ``@app.error``: per-call hook ``fn(exc, ctx) -> dict | None``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            self._ensure_event_dispatch()
            return self._events.error(tag)
        return self._decorate(HookKind.ERROR, fn, name, order)

    @property
    def events(self) -> EventRegistry:
        """The tag-keyed event registry backing the ``before/after/stream/
        error`` decorators. Read it for ``all()`` / ``describe()``."""
        return self._events

    def add_handler(self, handler: Any) -> Any:
        """Register an :class:`~margam.EventHandler` group (or any object
        whose methods are named ``{event}_{tag}``) and return it. Equivalent
        to decorating each method with the matching ``@app.<event>(tag)``."""
        self._ensure_event_dispatch()
        self._events.add(handler)
        return handler

    def _ensure_event_dispatch(self) -> None:
        """Wire the tag-event dispatchers into the hook pipeline (once)."""
        if self._events_wired:
            return
        self._events_wired = True

        @self.before(name="events:before")
        async def _dispatch_before(ctx):
            tags = ctx.state.get(STATE_KEY) or []
            if tags:
                await self._events.run_before(tags, ctx)

        @self.after(name="events:after")
        async def _dispatch_after(payload, ctx):
            tags = ctx.state.get(STATE_KEY) or []
            if tags:
                payload = await self._events.run_after(tags, payload, ctx)
            return payload

        @self.stream(name="events:stream")
        async def _dispatch_stream(chunk, ctx):
            tags = ctx.state.get(STATE_KEY) or []
            if not tags:
                return chunk
            scratch = ctx.state.setdefault(SCRATCH_KEY, {})
            return await self._events.run_stream(tags, chunk, ctx, scratch)

        @self.error(name="events:error")
        async def _dispatch_error(exc, ctx):
            tags = ctx.state.get(STATE_KEY) or []
            if tags:
                return await self._events.run_error(tags, exc, ctx)
            return None

    def _decorate(self, kind, fn, name, order):
        def deco(func):
            self._hooks.register(kind, func, name or getattr(func, "__name__", "hook"), order)
            return func

        return deco if fn is None else deco(fn)

# -- request pipeline ---------------------------------------------------

    _KIND_PREPARE = {
        "chat": "prepare_chat",
        "completions": "prepare_completions",
        "embeddings": "prepare_embeddings",
        "audio_transcriptions": "prepare_audio_transcriptions",
        "audio_translations": "prepare_audio_translations",
        "audio_speech": "prepare_audio_speech",
        "images_generations": "prepare_images_generations",
        "images_edits": "prepare_images_edits",
        "images_variations": "prepare_images_variations",
        "moderations": "prepare_moderations",
        "files": "prepare_files",
        "files_delete": "prepare_files_delete",
        "files_content": "prepare_files_content",
        "fine_tuning_jobs": "prepare_fine_tuning_jobs",
        "fine_tuning_jobs_list": "prepare_fine_tuning_jobs_list",
        "fine_tuning_jobs_cancel": "prepare_fine_tuning_jobs_cancel",
        "fine_tuning_events": "prepare_fine_tuning_events",
        "batches": "prepare_batches",
        "batches_retrieve": "prepare_batches_retrieve",
        "batches_cancel": "prepare_batches_cancel",
        "batches_list": "prepare_batches_list",
        "vector_stores": "prepare_vector_stores",
        "vector_stores_retrieve": "prepare_vector_stores_retrieve",
        "vector_stores_delete": "prepare_vector_stores_delete",
        "vector_stores_list": "prepare_vector_stores_list",
        "assistants": "prepare_assistants",
        "assistants_retrieve": "prepare_assistants_retrieve",
        "assistants_delete": "prepare_assistants_delete",
        "assistants_list": "prepare_assistants_list",
        "threads": "prepare_threads",
        "threads_retrieve": "prepare_threads_retrieve",
        "threads_delete": "prepare_threads_delete",
        "threads_messages": "prepare_threads_messages",
        "threads_messages_list": "prepare_threads_messages_list",
        "threads_messages_retrieve": "prepare_threads_messages_retrieve",
        "threads_runs": "prepare_threads_runs",
        "threads_runs_retrieve": "prepare_threads_runs_retrieve",
        "threads_runs_list": "prepare_threads_runs_list",
        "threads_runs_cancel": "prepare_threads_runs_cancel",
        "threads_runs_steps": "prepare_threads_runs_steps",
        "threads_runs_steps_list": "prepare_threads_runs_steps_list",
    }

    _KIND_PARSE = {
        "chat": "parse_response",
        "completions": "parse_completion_response",
        "embeddings": "parse_embeddings_response",
        "audio_transcriptions": "parse_audio_transcriptions_response",
        "audio_translations": "parse_audio_translations_response",
        "audio_speech": "parse_audio_speech_response",
        "images_generations": "parse_images_generations_response",
        "images_edits": "parse_images_edits_response",
        "images_variations": "parse_images_variations_response",
        "moderations": "parse_moderations_response",
        "files": "parse_files_response",
        "files_delete": "parse_files_delete_response",
        "files_content": "parse_files_content_response",
        "fine_tuning_jobs": "parse_fine_tuning_jobs_response",
        "fine_tuning_jobs_list": "parse_fine_tuning_jobs_list_response",
        "fine_tuning_jobs_cancel": "parse_fine_tuning_jobs_cancel_response",
        "fine_tuning_events": "parse_fine_tuning_events_response",
        "batches": "parse_batches_response",
        "batches_retrieve": "parse_batches_retrieve_response",
        "batches_cancel": "parse_batches_cancel_response",
        "batches_list": "parse_batches_list_response",
        "vector_stores": "parse_vector_stores_response",
        "vector_stores_retrieve": "parse_vector_stores_retrieve_response",
        "vector_stores_delete": "parse_vector_stores_delete_response",
        "vector_stores_list": "parse_vector_stores_list_response",
        "assistants": "parse_assistants_response",
        "assistants_retrieve": "parse_assistants_retrieve_response",
        "assistants_delete": "parse_assistants_delete_response",
        "assistants_list": "parse_assistants_list_response",
        "threads": "parse_threads_response",
        "threads_retrieve": "parse_threads_retrieve_response",
        "threads_delete": "parse_threads_delete_response",
        "threads_messages": "parse_threads_messages_response",
        "threads_messages_list": "parse_threads_messages_list_response",
        "threads_messages_retrieve": "parse_threads_messages_retrieve_response",
        "threads_runs": "parse_threads_runs_response",
        "threads_runs_retrieve": "parse_threads_runs_retrieve_response",
        "threads_runs_list": "parse_threads_runs_list_response",
        "threads_runs_cancel": "parse_threads_runs_cancel_response",
        "threads_runs_steps": "parse_threads_runs_steps_response",
        "threads_runs_steps_list": "parse_threads_runs_steps_list_response",
    }

    _KIND_PARSE_CHUNK = {
        "chat": "parse_chunk",
        "completions": "parse_completion_chunk",
    }

    _STREAMABLE_KINDS = {"chat", "completions"}

    async def _prepare(self, body: dict[str, Any], stream: bool, kind: str = "chat") -> tuple[RequestContext, Any, PreparedRequest]:
        """Run request hooks, resolve the route, and build the upstream request."""
        ctx = RequestContext(body=copy.deepcopy(body), stream=stream, config=self.config)
        for hook in self._hooks.requests():
            out = await maybe_await(hook.fn(ctx))
            if out is not None:
                ctx.body = out

        route = self.router.resolve(ctx.body.get("model") or "")
        ctx.provider, ctx.upstream_model, ctx.request_model = (
            route.provider,
            route.model,
            ctx.body.get("model"),
        )
        ctx.body["model"] = route.model
        provider = self.providers[route.provider]

        prepare_method = self._KIND_PREPARE.get(kind)
        if prepare_method is None:
            raise ApiError(400, f"Unknown endpoint kind: {kind}", error_type="invalid_request_error", param="model")
        prepare = getattr(provider, prepare_method)
        prepared = prepare(ctx)
        return ctx, provider, prepared

    async def _run_error_hooks(self, exc: Exception, ctx: RequestContext) -> dict[str, Any] | None:
        return await self._hooks.apply_error(exc, ctx)

    def _default_error_body(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ApiError):
            return exc.body
        return ApiError(500, f"Internal error: {exc}", error_type="server_error").body

    def _record(self, ctx: RequestContext, *, status: int = 200, error: str | None = None, usage: dict[str, Any] | None = None, source: str = "chat") -> CallRecord:
        usage = usage or {}
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        return self.telemetry.record(
            source=source,
            request_model=ctx.request_model,
            provider=ctx.provider,
            upstream_model=ctx.upstream_model,
            stream=ctx.stream,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cached_tokens=details.get("cached_tokens"),
            latency_ms=ctx.elapsed_ms(),
            status=status,
            error=error,
        )

    # -- non-streaming chat / completions -------------------------------------

    async def complete(self, body: dict[str, Any], *, kind: str = "chat") -> GatewayResponse:
        ctx: RequestContext | None = None
        try:
            ctx, provider, prepared = await self._prepare(body, stream=False, kind=kind)
            parse_method = self._KIND_PARSE.get(kind)
            if parse_method is None:
                raise ApiError(400, f"Unknown endpoint kind: {kind}", error_type="invalid_request_error", param="model")
            parse = getattr(provider, parse_method)
            resp = await self.transport.request(prepared)
            payload, status = parse(resp, ctx)
            payload = await self._hooks.apply_response(payload, ctx)
            self.telemetry.emit(self._record(ctx, status=status, usage=provider.extract_usage(payload), source=kind))
            return GatewayResponse(body=payload, status=status)
        except ApiError as exc:
            handled = await self._run_error_hooks(exc, ctx) if ctx else None
            self.telemetry.emit(self._record(ctx, status=exc.status, error=exc.message, source=kind) if ctx else self.telemetry.record(status=exc.status, error=exc.message, source=kind))
            if handled is not None:
                return GatewayResponse(body=handled, status=exc.status)
            return GatewayResponse(body=exc.body, status=exc.status)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unhandled error in wrapper.complete")
            handled = await self._run_error_hooks(exc, ctx) if ctx else None
            self.telemetry.emit(self._record(ctx, status=500, error=repr(exc), source=kind) if ctx else self.telemetry.record(status=500, error=repr(exc), source=kind))
            body = handled if handled is not None else self._default_error_body(exc)
            return GatewayResponse(body=body, status=500)

    # -- streaming chat / completions -----------------------------------------

    async def open_stream(self, body: dict[str, Any], *, kind: str = "chat") -> "StreamHandle":
        """Open the upstream stream. Raises :class:`ApiError` (or any other
        exception) *before* any SSE bytes would be sent, so the caller can
        still respond with a JSON error instead of a broken stream."""
        if kind not in self._STREAMABLE_KINDS:
            raise ApiError(400, f"Kind '{kind}' does not support streaming", error_type="invalid_request_error", param="model")
        ctx, provider, prepared = await self._prepare(body, stream=True, kind=kind)
        upstream = await self.transport.open_stream(prepared)
        if upstream.status >= 400:
            try:
                err_body = await upstream.json()
            except Exception:
                err_body = None
            raise ApiError.from_openai_body(err_body, implicit_status=upstream.status)
        return StreamHandle(self, ctx, provider, upstream, kind=kind)

    # -- model catalog ------------------------------------------------------

    @property
    def catalog_ttl(self) -> float:
        return self._catalog_ttl

    async def provider_models(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, provider in self.providers.items():
            configured = provider.configured_models()
            if configured:
                out[name] = configured
                continue
            try:
                discovered = await provider.list_models(self.transport)
            except Exception as exc:  # catalog must degrade gracefully
                logger.warning("model discovery failed for '%s': %s", name, exc)
                discovered = []
            out[name] = discovered
        return out

    async def models(self) -> list[dict[str, Any]]:
        if self._catalog_cache is not None and (time.monotonic() - self._catalog_at) < self.catalog_ttl:
            return self._catalog_cache
        provider_models = await self.provider_models()
        self._catalog_cache = self.router.catalog(provider_models)
        self._catalog_at = time.monotonic()
        return self._catalog_cache

    def invalidate_models_cache(self) -> None:
        """Drop the cached ``/v1/models`` catalog so the next call re-fetches."""
        self._catalog_cache = None
        self._catalog_at = 0.0


class StreamHandle:
    """Held-open upstream stream + the per-chunk hook pipeline.

    Constructed by :meth:`Wrapper.open_stream`; its :meth:`lines` generator
    is what an SSE transport feeds to the client.
    """

    def __init__(self, wrapper: Wrapper, ctx: RequestContext, provider: Any, upstream: UpstreamStream, *, kind: str = "chat") -> None:
        self._wrapper = wrapper
        self._ctx = ctx
        self._provider = provider
        self._upstream = upstream
        self._kind = kind
        parse_chunk_method = wrapper._KIND_PARSE_CHUNK.get(kind)
        if parse_chunk_method is None:
            raise ApiError(400, f"Kind '{kind}' does not support streaming", error_type="invalid_request_error", param="model")
        self._parse_chunk = getattr(provider, parse_chunk_method)
        self._last_usage: dict[str, Any] | None = None

    def lines(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            done = False
            error: str | None = None
            status = 200
            try:
                async for raw in self._upstream.lines():
                    chunk = self._parse_chunk(raw, self._ctx)
                    if chunk is None:
                        continue
                    if chunk is DONE:
                        yield _SSE_DONE
                        done = True
                        break
                    out = await self._wrapper._hooks.apply_stream(chunk, self._ctx)
                    if out is None:
                        continue
                    if isinstance(out, dict):
                        usage = self._provider.extract_usage(out)
                        if usage:
                            self._last_usage = usage
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                if not done:
                    yield _SSE_DONE
            except ApiError as exc:
                status = exc.status
                error = str(exc)
                body = (await self._wrapper._run_error_hooks(exc, self._ctx)) or exc.body
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                yield _SSE_DONE
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("stream failed")
                status = 500
                error = repr(exc)
                err = ApiError(500, f"Stream failed: {exc}", error_type="server_error")
                body = (await self._wrapper._run_error_hooks(exc, self._ctx)) or err.body
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                yield _SSE_DONE
            finally:
                self._wrapper.telemetry.emit(
                    self._wrapper._record(self._ctx, status=status, error=error, usage=self._last_usage, source=self._kind)
                )

        return _gen()


def _load_hook_module(entry: str, wrapper: Wrapper) -> None:
    module_name, sep, attr = entry.partition(":")
    module = importlib.import_module(module_name)
    if sep and attr:
        register = module
        for part in attr.split("."):
            register = getattr(register, part)
    else:
        register = getattr(module, "register", None)
        if register is None:
            raise ValueError(f"hook module '{entry}' has no register(wrapper) function or 'module:attr' form")
    result = register(wrapper)
    if isinstance(result, Wrapper) and result is not wrapper:
        raise ValueError(
            f"hook '{entry}' returned a different wrapper; pass the same wrapper instance or register in place"
        )


def _disabled_telemetry_config():
    from .config import TelemetryConfig

    return TelemetryConfig(enabled=False, emit="none")