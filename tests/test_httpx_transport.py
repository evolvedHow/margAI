"""Integration: the real HttpxTransport against httpx's MockTransport."""

from __future__ import annotations

import json

from margAI.config import GatewayConfig

from conftest import build_wrapper, chat_payload, chunk_payload, json_ok, sse_response


def test_request_through_httpx_transport():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o"  # namespace stripped upstream
        return json_ok(chat_payload("via httpx", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))

    import asyncio

    wrapper = build_wrapper(handler)
    result = asyncio.run(wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
    assert result.body["choices"][0]["message"]["content"] == "via httpx"
    assert result.body["usage"]["total_tokens"] == 2


def test_per_provider_timeout_reaches_httpx_request():
    from conftest import provider_config

    def handler(request):
        assert request.extensions["timeout"]["connect"] == 7.0  # from provider config
        return json_ok(chat_payload("timed"))

    import asyncio

    wrapper = build_wrapper(handler, providers=(provider_config(timeout=7.0),))
    result = asyncio.run(wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": []}))
    assert result.status == 200


def test_request_uses_client_default_when_no_provider_timeout():
    def handler(request):
        assert request.extensions["timeout"]["connect"] is None  # defer to client default
        return json_ok(chat_payload("ok"))

    import asyncio

    wrapper = build_wrapper(handler)
    result = asyncio.run(wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": []}))
    assert result.status == 200


def test_models_through_httpx_transport():
    def handler(request):
        return json_ok({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]})

    import asyncio

    wrapper = build_wrapper(handler)
    ids = [m["id"] for m in asyncio.run(wrapper.models())]
    assert "margAI/openai/gpt-4o" in ids


def test_models_falls_back_to_configured_on_discovery_failure():
    def handler(request):
        return json_ok({}, status=500)

    import asyncio

    wrapper = build_wrapper(handler)
    ids = [m["id"] for m in asyncio.run(wrapper.models())]
    # Discovery failed, so the configured list is used. The reserved dynamic id
    # is always advertised so clients can find it in a model dropdown.
    assert ids == ["margAI/dynamic", "margAI/openai/gpt-4o", "margAI/openai/gpt-4o-mini"]


def test_stream_through_httpx_transport():
    chunks = [chunk_payload("one"), chunk_payload("two", finish_reason="stop")]

    def handler(request):
        return sse_response(chunks)

    import asyncio

    wrapper = build_wrapper(handler)

    async def go():
        handle = await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": [], "stream": True})
        return [line async for line in handle.lines()]

    lines = asyncio.run(go())
    payloads = [json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: ") and not l.startswith("data: [DONE]")]
    assert [p["choices"][0]["delta"].get("content") for p in payloads] == ["one", "two"]
    assert lines[-1] == "data: [DONE]\n\n"


def test_stream_error_before_first_byte():
    def handler(request):
        return json_ok({"error": {"message": "nope", "type": "invalid_request_error"}}, status=400)

    import asyncio

    from margAI import ApiError

    wrapper = build_wrapper(handler)
    try:
        asyncio.run(wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": [], "stream": True}))
        assert False, "expected ApiError"
    except ApiError as exc:
        assert exc.status == 400
        assert exc.message == "nope"