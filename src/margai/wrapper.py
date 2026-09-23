"""The :class:`Wrapper` -- interceptors around any upstream provider.

A wrapper is the thing you design around:

    wrapper = Wrapper(providers={...}, transport=...)

    @wrapper.request
    async def decorate(ctx):
        # inspect / mutate ctx.body and ctx.state before upstream

    @wrapper.response
    def shape(payload, ctx):
        return payload

Non-streaming calls go through ``request`` -> transport -> ``response``.
Streaming calls go through ``request`` -> transport -> per-chunk ``stream``.
Any failure passes through ``error`` hooks before becoming an OpenAI-shaped
JSON error.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .config import Config
from .core import DONE
from .core.context import RequestContext
from .core.errors import ApiError
from .core.hooks import HookKind, HookRegistry, maybe_await
from .core.protocol import PreparedRequest, Transport, UpstreamStream
from .core.router import ModelRouter
from .providers import build_providers
from .telemetry import CallRecord, Telemetry

__all__ = ["GatewayResponse", "StreamHandle", "Wrapper"]

logger = logging.getLogger("margai.wrapper")

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
        prefix: str = "marg",
        expose: str = "prefixed",
        default_provider: str | None = None,
        telemetry: Telemetry | None = None,
        config: Config | None = None,
        name: str = "margai",
    ) -> None:
        self.name = name
        self.providers = providers
        self.transport = transport
        self.config = config
        self._hooks = HookRegistry()
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
        name: str = "margai",
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

    # -- hook decorators ----------------------------------------------------

    def request(self, fn=None, *, name: str | None = None, order: int = 0):
        return self._decorate(HookKind.REQUEST, fn, name, order)

    def response(self, fn=None, *, name: str | None = None, order: int = 0):
        return self._decorate(HookKind.RESPONSE, fn, name, order)

    def stream(self, fn=None, *, name: str | None = None, order: int = 0):
        return self._decorate(HookKind.STREAM, fn, name, order)

    def error(self, fn=None, *, name: str | None = None, order: int = 0):
        return self._decorate(HookKind.ERROR, fn, name, order)

    def _decorate(self, kind, fn, name, order):
        def deco(func):
            self._hooks.register(kind, func, name or getattr(func, "__name__", "hook"), order)
            return func

        return deco if fn is None else deco(fn)

    # -- request pipeline ---------------------------------------------------

    async def _prepare(self, body: dict[str, Any], stream: bool) -> tuple[RequestContext, Any, PreparedRequest]:
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
        prepared = provider.prepare_chat(ctx)
        return ctx, provider, prepared

    async def _run_error_hooks(self, exc: Exception, ctx: RequestContext) -> dict[str, Any] | None:
        return await self._hooks.apply_error(exc, ctx)

    def _default_error_body(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ApiError):
            return exc.body
        return ApiError(500, f"Internal error: {exc}", error_type="server_error").body

    def _record(self, ctx: RequestContext, *, status: int = 200, error: str | None = None, usage: dict[str, Any] | None = None) -> CallRecord:
        usage = usage or {}
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        return self.telemetry.record(
            source="chat",
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

    # -- non-streaming chat -------------------------------------------------

    async def complete(self, body: dict[str, Any]) -> GatewayResponse:
        ctx: RequestContext | None = None
        try:
            ctx, provider, prepared = await self._prepare(body, stream=False)
            resp = await self.transport.request(prepared)
            payload, status = provider.parse_response(resp, ctx)
            payload = await self._hooks.apply_response(payload, ctx)
            self.telemetry.emit(self._record(ctx, status=status, usage=provider.extract_usage(payload)))
            return GatewayResponse(body=payload, status=status)
        except ApiError as exc:
            handled = await self._run_error_hooks(exc, ctx) if ctx else None
            self.telemetry.emit(self._record(ctx, status=exc.status, error=exc.message) if ctx else self.telemetry.record(status=exc.status, error=exc.message))
            if handled is not None:
                return GatewayResponse(body=handled, status=exc.status)
            return GatewayResponse(body=exc.body, status=exc.status)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unhandled error in wrapper.complete")
            handled = await self._run_error_hooks(exc, ctx) if ctx else None
            self.telemetry.emit(self._record(ctx, status=500, error=repr(exc)) if ctx else self.telemetry.record(status=500, error=repr(exc)))
            body = handled if handled is not None else self._default_error_body(exc)
            return GatewayResponse(body=body, status=500)

    # -- streaming chat -----------------------------------------------------

    async def open_stream(self, body: dict[str, Any]) -> "StreamHandle":
        """Open the upstream stream. Raises :class:`ApiError` (or any other
        exception) *before* any SSE bytes would be sent, so the caller can
        still respond with a JSON error instead of a broken stream."""
        ctx, provider, prepared = await self._prepare(body, stream=True)
        upstream = await self.transport.open_stream(prepared)
        if upstream.status >= 400:
            try:
                err_body = await upstream.json()
            except Exception:
                err_body = None
            raise ApiError.from_openai_body(err_body, implicit_status=upstream.status)
        return StreamHandle(self, ctx, provider, upstream)

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


class StreamHandle:
    """Held-open upstream stream + the per-chunk hook pipeline.

    Constructed by :meth:`Wrapper.open_stream`; its :meth:`lines` generator
    is what an SSE transport feeds to the client.
    """

    def __init__(self, wrapper: Wrapper, ctx: RequestContext, provider: Any, upstream: UpstreamStream) -> None:
        self._wrapper = wrapper
        self._ctx = ctx
        self._provider = provider
        self._upstream = upstream
        self._last_usage: dict[str, Any] | None = None

    def lines(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            done = False
            error: str | None = None
            status = 200
            try:
                async for raw in self._upstream.lines():
                    chunk = self._provider.parse_chunk(raw, self._ctx)
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
                    self._wrapper._record(self._ctx, status=status, error=error, usage=self._last_usage)
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