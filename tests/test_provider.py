"""OpenAI-compatible provider unit tests."""

from __future__ import annotations

import pytest
from conftest import FakeTransport, make_config, provider_config

from margAI.core import DONE, ApiError
from margAI.core.context import RequestContext
from margAI.core.protocol import PreparedRequest, UpstreamResponse
from margAI.providers import build_providers


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
    import asyncio

    p = provider()
    transport = FakeTransport(responses=[UpstreamResponse(500, {"error": {"message": "boom"}})])

    async def call() -> None:
        await p.list_models(transport)

    with pytest.raises(ApiError):
        asyncio.run(call())


def test_extract_usage():
    p = provider()
    assert p.extract_usage({"usage": {"total_tokens": 3}}) == {"total_tokens": 3}
    assert p.extract_usage({"choices": []}) == {}

# -- kinds a provider does not speak --------------------------------------


UNSUPPORTED_KINDS = [
    ("completions", "prepare_completions"),
    ("embeddings", "prepare_embeddings"),
    ("images_generations", "prepare_images_generations"),
    ("audio_transcriptions", "prepare_audio_transcriptions"),
]


def anthropic():
    cfg = provider_config(name="anthropic", kind="anthropic", base_url="https://api.anthropic.com/v1")
    return build_providers(make_config(providers=(cfg,)))["anthropic"]


@pytest.mark.parametrize(("kind", "method"), UNSUPPORTED_KINDS)
def test_provider_base_declares_a_404_fallback_for_every_kind(kind, method):
    """A kind in `Wrapper._KIND_PREPARE` that a provider has not implemented
    must degrade to 'that provider cannot do that', not to an AttributeError
    that surfaces as a redacted 500."""
    for p in (provider(), anthropic()):
        assert callable(getattr(p, method, None)), f"{p.name} has no {method}"


@pytest.mark.parametrize(("kind", "method"), UNSUPPORTED_KINDS)
def test_the_fallback_raises_a_404_naming_the_provider_and_endpoint(kind, method):
    p = anthropic()
    with pytest.raises(ApiError) as exc:
        getattr(p, method)(RequestContext(body={}))
    assert exc.value.status == 404
    assert "anthropic" in exc.value.message
    assert "/" in exc.value.message


@pytest.mark.parametrize(
    ("kind", "parse_method"),
    [
        ("completions", "parse_completion_response"),
        ("embeddings", "parse_embeddings_response"),
        ("images_generations", "parse_images_generations_response"),
        ("audio_transcriptions", "parse_audio_transcriptions_response"),
    ],
)
def test_the_parse_half_of_each_kind_also_falls_back_to_404(kind, parse_method):
    """Both halves of a kind's pair need the fallback, or the error lands one
    step later than the routing decision."""
    with pytest.raises(ApiError) as exc:
        getattr(anthropic(), parse_method)(UpstreamResponse(200, {}), RequestContext(body={}))
    assert exc.value.status == 404


# -- streamed token usage ---------------------------------------------------


def stream_body(**extra):
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], **extra}


def test_a_streaming_chat_asks_the_upstream_for_usage():
    """Without include_usage the final SSE chunk carries no counts, so a
    streamed call would be invisible to telemetry and to the ledger."""
    req = provider(api_key="sk").prepare_chat(RequestContext(body=stream_body(stream=True)))
    assert req.json["stream_options"] == {"include_usage": True}


def test_a_streaming_completion_asks_for_usage_too():
    req = provider(api_key="sk").prepare_completions(
        RequestContext(body={"model": "gpt-3.5-turbo-instruct", "prompt": "hi", "stream": True})
    )
    assert req.json["stream_options"] == {"include_usage": True}


def test_a_non_streaming_call_is_left_alone():
    req = provider(api_key="sk").prepare_chat(RequestContext(body=stream_body()))
    assert "stream_options" not in req.json


def test_an_explicit_include_usage_is_never_overwritten():
    body = stream_body(stream=True, stream_options={"include_usage": False})
    req = provider(api_key="sk").prepare_chat(RequestContext(body=body))
    assert req.json["stream_options"] == {"include_usage": False}


def test_existing_stream_options_are_preserved():
    body = stream_body(stream=True, stream_options={"foo": "bar"})
    req = provider(api_key="sk").prepare_chat(RequestContext(body=body))
    assert req.json["stream_options"] == {"foo": "bar", "include_usage": True}
