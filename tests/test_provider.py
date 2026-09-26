"""OpenAI-compatible provider unit tests."""

from __future__ import annotations

import pytest

from margAI.core import DONE, ApiError
from margAI.core.context import RequestContext
from margAI.core.protocol import PreparedRequest, UpstreamResponse
from margAI.providers import build_providers

from conftest import FakeTransport, make_config, provider_config


def provider(**kw):
    cfg = provider_config(**kw)
    return build_providers(make_config(providers=(cfg,)))["openai"]


def test_prepare_chat_builds_url_headers_and_body():
    p = provider(api_key="sk-abc", base_url="https://api.openai.com/v1/")
    ctx = RequestContext(body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5})
    req = p.prepare_chat(ctx)
    assert isinstance(req, PreparedRequest)
    assert req.url == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-abc"
    assert req.headers["Content-Type"] == "application/json"
    assert req.json["model"] == "gpt-4o"
    assert req.json["temperature"] == 0.5


def test_prepare_chat_omits_auth_without_key():
    p = provider(api_key=None)
    req = p.prepare_chat(RequestContext(body={"messages": []}))
    assert "Authorization" not in req.headers


def test_build_providers_resolves_api_key_env():
    from margAI.providers import build_providers

    cfg = make_config(providers=(provider_config(api_key=None, api_key_env="MARG_TEST_KEY"),))
    providers = build_providers(cfg, env={"MARG_TEST_KEY": "sk-from-env"})
    req = providers["openai"].prepare_chat(RequestContext(body={"messages": []}))
    assert req.headers["Authorization"] == "Bearer sk-from-env"


def test_build_providers_prefers_inline_api_key_over_env():
    from margAI.providers import build_providers

    cfg = make_config(
        providers=(provider_config(api_key="sk-inline", api_key_env="MARG_TEST_KEY"),)
    )
    providers = build_providers(cfg, env={"MARG_TEST_KEY": "sk-from-env"})
    req = providers["openai"].prepare_chat(RequestContext(body={"messages": []}))
    assert req.headers["Authorization"] == "Bearer sk-inline"


def test_build_providers_missing_env_sends_no_auth():
    from margAI.providers import build_providers

    cfg = make_config(providers=(provider_config(api_key=None, api_key_env="UNSET_VAR"),))
    providers = build_providers(cfg, env={})
    req = providers["openai"].prepare_chat(RequestContext(body={"messages": []}))
    assert "Authorization" not in req.headers


def test_extra_headers_merged():
    p = provider(api_key=None, extra={"headers": {"X-Custom": "1"}})
    req = p.prepare_chat(RequestContext(body={"messages": []}))
    assert req.headers["X-Custom"] == "1"


def test_prepare_carries_provider_timeout():
    p = provider(timeout=12.5)
    req = p.prepare_chat(RequestContext(body={"messages": []}))
    assert req.timeout == 12.5
    req = p.prepare_completions(RequestContext(body={"prompt": "p"}))
    assert req.timeout == 12.5


def test_prepare_timeout_none_by_default():
    p = provider()
    req = p.prepare_chat(RequestContext(body={"messages": []}))
    assert req.timeout is None  # transport default wins


def test_parse_response_success():
    p = provider()
    payload = {"id": "x", "choices": [], "usage": {"total_tokens": 3}}
    body, status = p.parse_response(UpstreamResponse(200, payload), None)
    assert status == 200
    assert body["usage"]["total_tokens"] == 3


def test_parse_response_error_raises_api_error():
    p = provider()
    with pytest.raises(ApiError) as e:
        p.parse_response(UpstreamResponse(429, {"error": {"message": "slow down", "type": "rate_limit_error"}}), None)
    assert e.value.status == 429
    assert e.value.message == "slow down"


def test_parse_response_non_json_body():
    p = provider()
    body, status = p.parse_response(UpstreamResponse(200, "not a dict"), None)
    assert status == 200
    assert body == {"choices": []}


def test_parse_chunk_framing():
    p = provider()
    assert p.parse_chunk("data: [DONE]", None) is DONE
    assert p.parse_chunk(": keepalive", None) is None
    assert p.parse_chunk("event: message", None) is None
    assert p.parse_chunk("", None) is None
    assert p.parse_chunk("data: notjson", None) is None
    chunk = p.parse_chunk('data: {"object": "chat.completion.chunk"}', None)
    assert chunk["object"] == "chat.completion.chunk"


def test_list_models_uses_transport():
    p = provider()
    transport = FakeTransport(responses=[UpstreamResponse(200, {"data": [{"id": "a"}, {"id": "b"}]})])

    async def go():
        return await p.list_models(transport)

    import asyncio

    assert asyncio.run(go()) == ["a", "b"]
    assert transport.requested[0].url.endswith("/models")


def test_list_models_raises_on_error():
    p = provider()
    transport = FakeTransport(responses=[UpstreamResponse(500, {"error": {"message": "boom"}})])

    with pytest.raises(ApiError):
        import asyncio

        asyncio.run(p.list_models(transport))


def test_extract_usage():
    p = provider()
    assert p.extract_usage({"usage": {"total_tokens": 3}}) == {"total_tokens": 3}
    assert p.extract_usage({"choices": []}) == {}