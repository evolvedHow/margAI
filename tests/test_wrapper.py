"""Wrapper end-to-end tests using the deterministic fake transport."""

from __future__ import annotations

import json

import pytest

from margai import ApiError
from margai.config import GatewayConfig, TelemetryConfig
from margai.core.protocol import UpstreamResponse
from margai.telemetry import Telemetry

from conftest import FakeTransport, chat_payload, make_wrapper, provider_config


def test_complete_passes_through_and_applies_hooks():
    seen = []

    async def handler(req):
        body = json.loads(json.dumps(req.json))
        seen.append(("http", body["model"]))
        return UpstreamResponse(status=200, body=chat_payload("ok", usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}))

    wrapper = make_wrapper(FakeTransport(responses=[handler]))

    @wrapper.request
    def add_system(ctx):
        ctx.add_system_prompt("be terse")
        seen.append(("request", [m["role"] for m in ctx.messages]))

    @wrapper.response
    def footer(payload, ctx):
        payload["choices"][0]["message"]["content"] += "!"
        seen.append(("response", ctx.upstream_model))
        return payload

    import asyncio

    result = asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
    assert result.status == 200
    content = result.body["choices"][0]["message"]["content"]
    assert content == "ok!"
    assert ("request", ["system", "user"]) in seen
    assert ("http", "gpt-4o") in seen  # model stripped of namespace before upstream
    assert ("response", "gpt-4o") in seen


def test_request_hook_can_replace_body():
    sent = []
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(200, chat_payload("x"))]))

    @wrapper.request
    def override(ctx):
        return {"model": "marg/openai/gpt-4o", "messages": [{"role": "system", "content": "replaced"}]}

    @wrapper.response
    def nothing(payload, ctx):
        return payload

    sent.clear()

    async def spy(req):
        sent.append(req.json)
        return UpstreamResponse(200, chat_payload("x"))

    wrapper.transport.responses.insert(0, spy)

    import asyncio

    asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
    assert sent[0]["messages"][0]["content"] == "replaced"


def test_upstream_error_becomes_openai_error_body():
    upstream = {"error": {"message": "rate limited", "type": "rate_limit_error", "code": "429"}}
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(status=429, body=upstream)]))
    import asyncio

    result = asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": []}))
    assert result.status == 429
    assert result.body["error"]["message"] == "rate limited"


def test_error_hook_can_take_over_body():
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(status=503, body={"error": {"message": "up"}})]))

    @wrapper.error
    def friendly(exc, ctx):
        return {"error": {"message": "please retry later", "type": "retry", "param": None, "code": None}}

    import asyncio

    result = asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": []}))
    assert result.status == 503
    assert result.body["error"]["message"] == "please retry later"


def test_provider_qualified_unknown_model_passes_through():
    """A provider-qualified id is never 404'd by the router: the upstream
    provider remains authoritative for model validation."""
    sent = []

    async def spy(req):
        sent.append(req.json["model"])
        return UpstreamResponse(200, chat_payload("ok"))

    wrapper = make_wrapper(FakeTransport(responses=[spy]))
    import asyncio

    result = asyncio.run(wrapper.complete({"model": "marg/openai/nope", "messages": []}))
    assert result.status == 200
    assert sent == ["nope"]


def test_telemetry_records_usage():
    records = []
    telemetry = Telemetry(TelemetryConfig(enabled=True, emit="callback", callback="irrelevant"), callback=records.append)
    wrapper = make_wrapper(
        FakeTransport(
            responses=[
                UpstreamResponse(
                    200,
                    chat_payload(
                        "ok",
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                            "prompt_tokens_details": {"cached_tokens": 3},
                        },
                    ),
                )
            ]
        ),
        telemetry=telemetry,
    )
    import asyncio

    asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": []}))
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "openai"
    assert rec.upstream_model == "gpt-4o"
    assert rec.prompt_tokens == 10
    assert rec.cached_tokens == 3
    assert rec.status == 200
    assert rec.error is None


def test_cost_estimate_when_configured():
    from margai.config import CostEntry, TelemetryConfig

    tele = Telemetry(TelemetryConfig(enabled=True, emit="none", costs=(CostEntry("openai", "gpt-4o", 2.5, 10.0),)))
    records = []

    class Swappable(Telemetry):
        def emit(self, record):
            record.cost_usd = self.cost_for(record)
            records.append(record)

    sw = Swappable(tele.config)
    wrapper = make_wrapper(
        FakeTransport(
            responses=[
                UpstreamResponse(
                    200,
                    chat_payload("ok", usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000}),
                )
            ]
        ),
        telemetry=sw,
    )
    import asyncio

    asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o", "messages": []}))
    assert records[0].cost_usd == pytest.approx(12.5)


def test_request_hook_runs_before_routing_is_visible():
    routes = []
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(200, chat_payload("x"))]))

    @wrapper.response
    def capture(payload, ctx):
        routes.append(ctx.upstream_model)
        return payload

    import asyncio

    asyncio.run(wrapper.complete({"model": "marg/openai/gpt-4o-mini", "messages": []}))
    assert routes == ["gpt-4o-mini"]