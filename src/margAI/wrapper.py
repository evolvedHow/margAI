"""The :class:`Wrapper` -- intercept everything around any upstream provider.

A wrapper is the thing you design around, FastAPI-style:

    wrapper = Wrapper(providers={...}, transport=...)

    @wrapper.before("route")        # tag-keyed event (fires for that tag)
    def rewrite(ctx, tag):
        if tag.value:
            ctx.body["model"] = tag.value

    @wrapper.after                  # per-call hook (every call)
    def shape(payload, ctx):
        return payload

Non-streaming calls go through ``before`` -> transport -> ``after``.
Streaming calls go through ``before`` -> transport -> per-chunk ``stream``.
Any failure passes through ``error`` hooks before becoming an OpenAI-shaped
JSON error.

Two layers of extension sit on top of that, and they compose:

- **hooks/events** -- the mechanism. Four phases, per-call or keyed by tag.
- **marglets** -- the unit. One named enhancement (all four phases plus
  metadata) activated per call by a bangtag, and optionally owning the
  routing decision for the calls it is active on.

Request order is: bangtag parse -> marglet/per-call ``before`` -> *route*
(including the dynamic chain) -> transport -> ``after``/``stream`` -> record.
Routing sits after ``before`` on purpose: that is what lets a marglet change
which model serves the call.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, ClassVar

from .config import Config
from .core import DONE
from .core.context import RequestContext
from .core.errors import ApiError
from .core.events import SCRATCH_KEY, EventRegistry
from .core.hooks import HookKind, HookRegistry, maybe_await
from .core.intent import RoutingIntent
from .core.marglets import ACTIVE_KEY, Marglet, MargletRegistry
from .core.protocol import PreparedRequest, Transport, UpstreamStream
from .core.router import ModelRouter, Route
from .providers import build_providers
from .routing import Routing
from .telemetry import CallRecord, Telemetry

__all__ = ["GatewayResponse", "StreamHandle", "Wrapper"]

logger = logging.getLogger("margAI.wrapper")

_SSE_DONE = "data: [DONE]\n\n"

# Where the tag-event dispatcher sits among request hooks. Bangtag parsing
# installs at -100, so this lands after it: tags exist by the time the
# tag-keyed handlers (and therefore every marglet) run.
_EVENT_ORDER = -50

_MARGLET_PHASES = ("before", "after", "stream", "error")


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
        dynamic_model: str = "dynamic",
    ) -> None:
        self.name = name
        self.providers = providers
        self.transport = transport
        self.config = config
        self._hooks = HookRegistry()
        self._events = EventRegistry()
        self.marglets = MargletRegistry()
        self.router = ModelRouter(
            providers,
            prefix=prefix,
            expose=expose,
            default_provider=default_provider,
            dynamic_model=dynamic_model,
        )
        self.routing = Routing(self.router, default_provider=default_provider)
        self.telemetry = telemetry or Telemetry(_disabled_telemetry_config())
        self._catalog_cache: list[dict[str, Any]] | None = None
        self._catalog_at = 0.0
        self._catalog_ttl = config.gateway.models_cache_ttl if config else 300.0
        # Wired here, not lazily on first use: registration order must not
        # decide where the event dispatchers sit relative to user hooks.
        self._wire_event_dispatch()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        transport: Transport | None = None,
        *,
        load_hooks: bool = True,
        name: str = "margAI",
    ) -> Wrapper:
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
            dynamic_model=config.gateway.dynamic_model,
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
    # every call carrying that tag (see `ctx.tags`). `name=`,
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
        self._events.add(handler)
        return handler
    def _wire_event_dispatch(self) -> None:
        """Wire the tag-event dispatchers into the hook pipeline (once, from
        ``__init__``, so registration order cannot move them).

        Each dispatcher records the marglets the call activated before running
        the tag-keyed handlers, so ``ctx.marglets`` is populated for every
        phase (including the error phase, where it is the only way a handler
        can tell what was asked for).
        """

        @self.before(name="events:before", order=_EVENT_ORDER)
        async def _dispatch_before(ctx):
            tags = self._dispatch_tags(ctx)
            if not tags:
                return
            if self.marglets:
                ctx.state[ACTIVE_KEY] = [m.name for m in self.marglets.active(tags)]
            await self._events.run_before(tags, ctx)

        @self.after(name="events:after")
        async def _dispatch_after(payload, ctx):
            tags = self._dispatch_tags(ctx)
            if not tags:
                return payload
            return await self._events.run_after(tags, payload, ctx)

        @self.stream(name="events:stream")
        async def _dispatch_stream(chunk, ctx):
            tags = self._dispatch_tags(ctx)
            if not tags:
                return chunk
            scratch = ctx.state.setdefault(SCRATCH_KEY, {})
            return await self._events.run_stream(tags, chunk, ctx, scratch)

        @self.error(name="events:error")
        async def _dispatch_error(exc, ctx):
            tags = self._dispatch_tags(ctx)
            if not tags:
                return None
            return await self._events.run_error(tags, exc, ctx)

    def _dispatch_tags(self, ctx: RequestContext) -> list[Any]:
        """The tags to dispatch this phase, in *handler* order.

        Tag-keyed events run in the order the tags were written, which for
        marglets would make the user decide execution order by reordering
        their bangtag. Marglets declare an ``order`` and the registry resolves
        ties by name, so that is what wins here: marglet tags first, in
        registry order, then any plain event tags in their original order.

        With no marglets registered this is the identity, so tag-keyed events
        behave exactly as they did before marglets existed.
        """
        tags = ctx.tags
        if not tags or not len(self.marglets):
            return tags
        rank = {m.name: i for i, m in enumerate(self.marglets.ordered())}
        marglet_tags = [t for t in tags if getattr(t, "name", None) in rank]
        plain_tags = [t for t in tags if getattr(t, "name", None) not in rank]
        return sorted(marglet_tags, key=lambda t: rank[getattr(t, "name", "")]) + plain_tags

    def _decorate(self, kind, fn, name, order):
        def deco(func):
            self._hooks.register(kind, func, name or getattr(func, "__name__", "hook"), order)
            return func

        return deco if fn is None else deco(fn)

    # -- marglets ------------------------------------------------------------

    def add_marglet(self, marglet: Marglet, *, replace: bool = False) -> Marglet:
        """Register a marglet and wire its phases onto the event registry.

        A marglet's hooks *are* tag-keyed event handlers under the hood, so
        registering one and decorating ``@app.before("name")`` are the same
        mechanism -- which is why the two can never disagree about ordering.
        """
        self.marglets.add(marglet, replace=replace)
        tag = marglet.name
        for phase, hook in marglet.phases.items():
            if hook is None:
                continue
            getattr(self, phase)(tag=tag)(hook)
        if marglet.select is not None:
            # A marglet that can route gets a selector scoped to the calls it
            # is active on, at the front of the chain so an explicit marglet
            # beats a generic policy.
            self.routing.add_selector(
                marglet.select, name=f"marglet:{marglet.name}", order=marglet.order, only_for=[marglet.name]
            )
        return marglet

    def marglet(self, fn=None, *, name=None, order=0, **kwargs):
        """Decorator form of :meth:`add_marglet`, with the same two modes as
        the phase decorators::

            @app.marglet("terse", summary="Answer briefly")
            def terse(ctx, tag):
                ctx.add_system_prompt("Answer in as few words as possible.")

            @app.marglet(summary="...")          # name from the function
            def terse(ctx, tag): ...

        Other phases go through ``Marglet(before=..., after=...)`` or
        :meth:`add_marglet_group`.
        """
        if isinstance(fn, str):
            name = name or fn
            fn = None
        if fn is not None:
            self.add_marglet(Marglet(name or fn.__name__, before=fn, order=order, **kwargs))
            return fn

        def deco(func):
            self.add_marglet(Marglet(name or func.__name__, before=func, order=order, **kwargs))
            return func

        return deco

    def add_marglet_group(self, group: Any) -> Any:
        """Register a group of marglets from ``{name}_{phase}``-style methods.

        The marglet-level analogue of :meth:`add_handler`: one object holding
        several related marglets, e.g.::

            class Bullets:
                def terse_before(self, ctx, tag): ...
                def terse_after(self, payload, ctx, tag): ...
                def bullets_before(self, ctx, tag): ...

            app.add_marglet_group(Bullets())

        The name is the part *before* the phase, matching ``add_handler``'s
        ``{event}_{name}`` shape transposed into ``{name}_{phase}``.
        """
        found: dict[str, dict[str, Any]] = {}
        for attr in dir(group):
            if attr.startswith("_") or "_" not in attr:
                continue
            name, _, phase = attr.rpartition("_")
            if not name or phase not in _MARGLET_PHASES:
                continue
            fn = getattr(group, attr)
            if callable(fn):
                found.setdefault(name, {})[phase] = fn
        for marglet_name, phases in found.items():
            self.add_marglet(Marglet(marglet_name, **phases))
        return group

    # -- request pipeline ---------------------------------------------------

    # The routed surface. Every kind here is backed by a real Provider method
    # and covered by tests; anything absent is not routable. See
    # docs/ENDPOINTS.md before adding one.
    _KIND_PREPARE: ClassVar[dict[str, str]] = {
        "chat": "prepare_chat",
        "completions": "prepare_completions",
        "embeddings": "prepare_embeddings",
        "images_generations": "prepare_images_generations",
        "audio_transcriptions": "prepare_audio_transcriptions",
    }

    _KIND_PARSE: ClassVar[dict[str, str]] = {
        "chat": "parse_response",
        "completions": "parse_completion_response",
        "embeddings": "parse_embeddings_response",
        "images_generations": "parse_images_generations_response",
        "audio_transcriptions": "parse_audio_transcriptions_response",
    }

    _KIND_PARSE_CHUNK: ClassVar[dict[str, str]] = {
        "chat": "parse_chunk",
        "completions": "parse_completion_chunk",
    }

    _STREAMABLE_KINDS: ClassVar[set[str]] = {"chat", "completions"}

    async def _prepare(
        self, body: dict[str, Any], stream: bool, kind: str = "chat"
    ) -> tuple[RequestContext, Any, PreparedRequest]:
        """Run request hooks, resolve the route, and build the upstream request.

        The body is shallow-copied, not deep-copied: a chat body is a list of
        messages that can be tens of kilobytes, and ``deepcopy`` on every
        request showed up as real latency. Hooks get the top-level dict to
        replace (``return {...}``) and mutate ``ctx.body`` in place; the
        nested message list is shared with the caller's parsed JSON, which is
        exactly what the pre-existing in-place hook contract already assumed.
        """
        ctx = RequestContext(body=dict(body), stream=stream, config=self.config)
        try:
            for hook in self._hooks.requests():
                out = await maybe_await(hook.fn(ctx))
                if out is not None:
                    ctx.body = out

            route = self._resolve(ctx)
            ctx.request_model = ctx.body.get("model")
            ctx.provider, ctx.upstream_model = route.provider, route.model
            ctx.body["model"] = route.model
            provider = self.providers[route.provider]

            prepare_method = self._KIND_PREPARE.get(kind)
            if prepare_method is None:
                raise ApiError(400, f"Unknown endpoint kind: {kind}", error_type="invalid_request_error", param="model")
            prepared = getattr(provider, prepare_method)(ctx)
        except BaseException as exc:
            # Hand the context to the error so hooks and telemetry can report
            # the call that failed, not just that something did. This covers
            # ordinary exceptions too: a before-hook that raises is the most
            # common failure of all, and it must still reach the error phase.
            with suppress(Exception):  # exotic exception types may forbid attrs
                exc.ctx = ctx  # type: ignore[attr-defined]
            raise
        return ctx, provider, prepared

    def _resolve(self, ctx: RequestContext) -> Route:
        """Route the call. Runs *after* every ``before`` hook, which is what
        lets a marglet steer routing.

        ``margAI/dynamic`` (or the bare ``dynamic`` when nothing claims it)
        hands off to the selector chain; everything else is a name lookup.
        """
        requested = ctx.body.get("model") or ""
        if not self.router.is_dynamic(requested):
            return self.router.resolve(requested)

        ctx.state["marglets.dynamic"] = True
        intent: RoutingIntent = ctx.intent
        # A bangtag that already pinned a concrete model wins over the chain:
        # "!margAI: dynamic, route=local/qwen" asked for a specific target.
        if intent.provider and intent.model:
            pinned = self.router.try_resolve(self.router.namespaced_id(intent.provider, intent.model))
            if pinned is not None:
                return Route(pinned.provider, pinned.model, "bangtag:route")
        route = self.routing.select(intent, ctx)
        ctx.state["marglets.route_reason"] = route.reason
        return route

    async def _run_error_hooks(self, exc: Exception, ctx: RequestContext | None) -> dict[str, Any] | None:
        """Run ``error`` hooks for a failure, if there is a context to run
        them against. Prefers the context attached to the error, because the
        local one is still ``None`` when ``_prepare`` itself failed."""
        ctx = getattr(exc, "ctx", None) or ctx
        if ctx is None:
            return None
        return await self._hooks.apply_error(exc, ctx)
    def _default_error_body(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ApiError):
            return exc.body
        # Never echo the exception text to the client; it leaks paths, keys
        # and upstream internals. The detail is in the log, where it belongs.
        logger.error("unhandled error: %r", exc, exc_info=exc)
        return ApiError(500, "Internal server error", error_type="server_error").body

    def _record(
        self,
        ctx: RequestContext,
        *,
        status: int = 200,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        source: str = "chat",
    ) -> CallRecord:
        tokens: dict[str, Any] = usage or {}
        raw_details = tokens.get("prompt_tokens_details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        return self.telemetry.record(
            source=source,
            request_model=ctx.request_model,
            provider=ctx.provider,
            upstream_model=ctx.upstream_model,
            stream=ctx.stream,
            prompt_tokens=tokens.get("prompt_tokens"),
            completion_tokens=tokens.get("completion_tokens"),
            total_tokens=tokens.get("total_tokens"),
            cached_tokens=details.get("cached_tokens"),
            latency_ms=ctx.elapsed_ms(),
            status=status,
            error=error,
            marglets=tuple(ctx.marglets),
            route_reason=ctx.state.get("marglets.route_reason") or "",
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
            return await self._fail(exc, ctx, kind, exc.status, exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unhandled error in wrapper.complete")
            return await self._fail(exc, ctx, kind, 500, repr(exc))

    async def _fail(
        self, exc: Exception, ctx: RequestContext | None, kind: str, status: int, error: str
    ) -> GatewayResponse:
        """The single failure path: error hooks, then telemetry, then a body.

        Shared by the non-streaming and streaming-preflight paths. They were
        near-identical but not identical -- one tested `ctx` for truthiness and
        the other for `is not None` -- so a failure could be recorded in one
        place and not the other as the two drifted.
        """
        handled = await self._run_error_hooks(exc, ctx)
        ctx = getattr(exc, "ctx", None) or ctx
        self.telemetry.emit(
            self._record(ctx, status=status, error=error, source=kind)
            if ctx is not None
            else self.telemetry.record(status=status, error=error, source=kind)
        )
        return GatewayResponse(body=handled if handled is not None else self._default_error_body(exc), status=status)

    # -- streaming chat / completions -----------------------------------------

    async def open_stream(self, body: dict[str, Any], *, kind: str = "chat") -> StreamHandle:
        """Open the upstream stream. Raises :class:`ApiError` (or any other
        exception) *before* any SSE bytes would be sent, so the caller can
        still respond with a JSON error instead of a broken stream.

        Failures here go through the same ``error`` hooks and telemetry as the
        non-streaming path. They used to skip both, which meant a rejected
        dynamic route or an upstream 401 looked like a call that never
        happened.
        """
        if kind not in self._STREAMABLE_KINDS:
            raise ApiError(
                400,
                f"Kind '{kind}' does not support streaming",
                error_type="invalid_request_error",
                param="model",
            )
        ctx: RequestContext | None = None
        try:
            ctx, provider, prepared = await self._prepare(body, stream=True, kind=kind)
            upstream = await self.transport.open_stream(prepared)
            if upstream.status >= 400:
                try:
                    err_body = await upstream.json()
                except Exception:
                    err_body = None
                raise ApiError.from_openai_body(err_body, implicit_status=upstream.status)
        except ApiError as exc:
            handled = await self._record_stream_failure(exc, exc.status, exc.message, ctx, kind)
            if handled is not None:
                # `with_body` returns a new ApiError, so this must be an
                # explicit `raise exc` -- a bare `raise` would re-raise the
                # original and drop the hook's body.
                exc = exc.with_body(handled)  # an error hook took over the body
            raise exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("failed to open stream")
            await self._record_stream_failure(exc, 500, repr(exc), ctx, kind)
            raise
        return StreamHandle(self, ctx, provider, upstream, kind=kind)

    async def _record_stream_failure(
        self, exc: Exception, status: int, error: str, ctx: RequestContext | None, kind: str
    ) -> dict[str, Any] | None:
        """Run the error hooks and record the failure for a stream that never
        produced a byte. Returns the error body if a hook took it over."""
        handled = await self._run_error_hooks(exc, ctx)
        ctx = getattr(exc, "ctx", None) or ctx
        self.telemetry.emit(
            self._record(ctx, status=status, error=error, source=kind)
            if ctx is not None
            else self.telemetry.record(status=status, error=error, source=kind)
        )
        return handled

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
        # Always advertise the dynamic id: a client picking from a dropdown
        # (LibreChat) can only use it if it can see it.
        self._catalog_cache = self.router.catalog(provider_models, include_dynamic=True)
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

    def __init__(
        self,
        wrapper: Wrapper,
        ctx: RequestContext,
        provider: Any,
        upstream: UpstreamStream,
        *,
        kind: str = "chat",
    ) -> None:
        self._wrapper = wrapper
        self._ctx = ctx
        self._provider = provider
        self._upstream = upstream
        self._kind = kind
        parse_chunk_method = wrapper._KIND_PARSE_CHUNK.get(kind)
        if parse_chunk_method is None:
            raise ApiError(
                400,
                f"Kind '{kind}' does not support streaming",
                error_type="invalid_request_error",
                param="model",
            )
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
                    self._wrapper._record(
                        self._ctx, status=status, error=error, usage=self._last_usage, source=self._kind
                    )
                )

        return _gen()


def _load_hook_module(entry: str, wrapper: Wrapper) -> None:
    module_name, sep, attr = entry.partition(":")
    module = importlib.import_module(module_name)
    # Either a module (`register` attribute) or a resolved dotted path; both
    # are dynamic, so the callable is only known at runtime.
    register: Any = module
    if sep and attr:
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