"""Shared test fixtures and wrapper builders."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx

from margAI import Wrapper
from margAI.config import (
    BillingConfig,
    Config,
    GatewayConfig,
    HookConfig,
    ProviderConfig,
    TelemetryConfig,
)
from margAI.core.protocol import PreparedRequest, UpstreamResponse
from margAI.providers import build_providers
from margAI.telemetry import Telemetry
from margAI.transport.httpx import HttpxTransport


def provider_config(
    name: str = "openai",
    *,
    kind: str = "openai",
    base_url: str = "https://upstream.test/v1",
    models: tuple[str, ...] = ("gpt-4o", "gpt-4o-mini"),
    api_key: str | None = "sk-test",
    api_key_env: str | None = None,
    default_model: str | None = None,
    timeout: float | None = None,
    extra: dict | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        models=models,
        default_model=default_model,
        timeout=timeout,
        extra=extra or {},
    )


def make_config(
    providers: tuple[ProviderConfig, ...] = (provider_config(),),
    gateway: GatewayConfig | None = None,
    hooks: HookConfig | None = None,
    telemetry: TelemetryConfig | None = None,
    models: dict | None = None,
    packs: dict | None = None,
    billing: BillingConfig | None = None,
) -> Config:
    return Config(
        gateway=gateway or GatewayConfig(),
        providers=providers,
        telemetry=telemetry or TelemetryConfig(enabled=False),
        billing=billing or BillingConfig(),
        hooks=hooks or HookConfig(),
        models=models or {},
        packs=packs or {},
    )


def make_wrapper(
    transport,
    *,
    providers: tuple[ProviderConfig, ...] = (provider_config(),),
    gateway: GatewayConfig | None = None,
    telemetry: Telemetry | None = None,
    models: dict | None = None,
    costs: tuple = (),
    config: Config | None = None,
    billing: BillingConfig | None = None,
) -> Wrapper:
    """Build a wrapper around a given transport (fake or httpx-mock)."""
    if config is None:
        base = telemetry.config if telemetry is not None else TelemetryConfig(enabled=False)
        config = make_config(
            providers=providers,
            gateway=gateway,
            telemetry=TelemetryConfig(
                enabled=base.enabled,
                emit=base.emit,
                callback=base.callback,
                costs=costs or base.costs,
            ),
            models=models,
            billing=billing,
        )
    gateway = config.gateway
    return Wrapper(
        providers=build_providers(config),
        transport=transport,
        prefix=gateway.prefix,
        expose=gateway.expose,
        default_provider=gateway.default_provider,
        telemetry=telemetry or Telemetry(TelemetryConfig(enabled=False)),
        config=config,
    )


# --------------------------------------------------------------------------
# httpx-driven wrapper (integration with the real transport)
# -------------------------------------------------------------------------


def build_wrapper(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    providers: tuple[ProviderConfig, ...] = (provider_config(),),
    gateway: GatewayConfig | None = None,
    telemetry: Telemetry | None = None,
) -> Wrapper:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return make_wrapper(
        HttpxTransport(client=client), providers=providers, gateway=gateway, telemetry=telemetry
    )


def json_ok(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def sse_response(chunks: list[dict], done: bool = True) -> httpx.Response:
    text_parts = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        text_parts.append("data: [DONE]\n\n")
    return httpx.Response(200, content="".join(text_parts).encode(), headers={"content-type": "text/event-stream"})


CHAT_MODEL = "margAI/openai/gpt-4o"

SSE_DONE = "data: [DONE]"


def sse_line(payload: dict) -> str:
    """One `data:` frame, for feeding FakeStream directly."""
    return f"data: {json.dumps(payload)}"


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def chat_body(*messages: dict, model: str = CHAT_MODEL, **extra) -> dict:
    """A request body for the chat surface, with sane defaults.

    Exists so the tests read as `chat_body(user("hi"))` instead of a
    `{"model": ..., "messages": [...]}` literal on nearly every line.
    """
    return {"model": model, "messages": list(messages) or [user("hi")], **extra}


def token_usage(prompt: int = 3, completion: int = 4, total: int | None = None) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total if total is not None else prompt + completion,
    }


def chat_payload(text: str = "hello", *, usage: dict | None = None, model: str = "echo") -> dict:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def chunk_payload(text: str, *, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "echo",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}],
    }


def completion_payload(text: str = "done", *, usage: dict | None = None, model: str = "echo") -> dict:
    payload = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def completion_chunk_payload(text: str = "", *, finish_reason: str | None = None) -> dict:
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1,
        "model": "echo",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


# --------------------------------------------------------------------------
# Deterministic fake transport (no HTTP at all)
# -------------------------------------------------------------------------


class FakeStream:
    def __init__(self, lines: list[str], status: int = 200, error_json=None) -> None:
        self._lines = lines
        self.status = status
        self._error_json = error_json

    async def json(self):
        return self._error_json

    def lines(self) -> AsyncIterator[str]:
        async def _gen():
            for line in self._lines:
                yield line

        return _gen()


class FakeTransport:
    """Scripted transport. ``responses`` is a list served in order for
    ``request``; ``streams`` likewise for ``open_stream``. Callables may also
    be given (receiving the PreparedRequest) for inspection."""

    def __init__(self, responses=None, streams=None) -> None:
        self.responses = list(responses or [])
        self.streams = list(streams or [])
        self.requested: list[PreparedRequest] = []

    async def request(self, req: PreparedRequest) -> UpstreamResponse:
        self.requested.append(req)
        item = self.responses.pop(0) if self.responses else None
        if callable(item):
            item = item(req)
            if hasattr(item, "__await__"):
                item = await item
            return item
        if item is None:
            return UpstreamResponse(status=200, body={})
        return item

    async def open_stream(self, req: PreparedRequest):
        self.requested.append(req)
        item = self.streams.pop(0) if self.streams else None
        if callable(item):
            item = item(req)
            if hasattr(item, "__await__"):
                item = await item
            return item
        if item is None:
            return FakeStream([], status=200)
        return item