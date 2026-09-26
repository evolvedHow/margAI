"""Anthropic provider unit tests (OpenAI <-> Anthropic translation)."""

from __future__ import annotations

import pytest
from conftest import FakeTransport, make_config, provider_config

from margAI.core import DONE, ApiError
from margAI.core.context import RequestContext
from margAI.core.protocol import UpstreamResponse
from margAI.providers import build_providers


def anthropic(**kw):
    cfg = provider_config(
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        name="anthropic",
        **kw,
    )
    return build_providers(make_config(providers=(cfg,)), env={})["anthropic"]


def ctx(body=None, *, stream=False):
    c = RequestContext(body=body or {}, stream=stream)
    c.upstream_model = "claude-sonnet-4-5"
    return c


def test_prepare_chat_translates_openai_shape():
    p = anthropic()
    body = {
        "model": "marg/anthropic/claude-sonnet-4-5",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        "stop": ["\n"],
        "temperature": 0.3,
        "top_p": 0.9,
        "user": "u-1",
    }
    req = p.prepare_chat(ctx(body))
    assert req.url == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-ant-test"
    assert req.headers["anthropic-version"] == "2023-06-01"
    assert req.headers["Content-Type"] == "application/json"
    j = req.json
    assert j["model"] == "claude-sonnet-4-5"
    assert j["system"] == "be terse"
    assert j["messages"] == [{"role": "user", "content": "hi"}]
    assert j["max_tokens"] == 1024
    assert j["stop_sequences"] == ["\n"]
    assert "stop" not in j
    assert j["temperature"] == 0.3
    assert j["top_p"] == 0.9
    assert j["metadata"] == {"user_id": "u-1"}
    assert "stream" not in j


def test_prepare_chat_stream_flag_and_custom_max_tokens():
    p = anthropic(extra={"max_tokens": 4096})
    req = p.prepare_chat(ctx({"messages": [{"role": "user", "content": "x"}]}, stream=True))
    assert req.json["stream"] is True
    assert req.json["max_tokens"] == 4096


def test_prepare_chat_requires_no_system_and_defaults_prompt():
    p = anthropic()
    req = p.prepare_chat(ctx({"messages": [{"role": "assistant", "content": "lead"}]}))
    # non-user/assistant/system roles dropped; empty user message synthesized
    assert req.json["messages"][0]["role"] == "user"


def test_prepare_carries_provider_timeout():
    p = anthropic(timeout=9.0)
    req = p.prepare_chat(ctx({"messages": []}))
    assert req.timeout == 9.0


def test_parse_response_maps_anthropic_to_chat():
    p = anthropic()
    body = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "model": "claude-sonnet-4-5",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    payload, status = p.parse_response(UpstreamResponse(200, body), ctx())
    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello!"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_parse_response_error_raises_with_nested_message():
    p = anthropic()
    err_body = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}}
    with pytest.raises(ApiError) as e:
        p.parse_response(UpstreamResponse(400, err_body), ctx())
    assert e.value.status == 400
    assert e.value.message == "bad request"


def test_parse_chunk_text_delta():
    p = anthropic()
    raw = 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"He"}}'
    chunk = p.parse_chunk(raw, ctx())
    assert chunk["choices"][0]["delta"]["content"] == "He"
    assert chunk["object"] == "chat.completion.chunk"


def test_parse_chunk_ignores_non_text_deltas_and_control_events():
    p = anthropic()
    raw = 'data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"}}'
    assert p.parse_chunk(raw, ctx()) is None
    assert p.parse_chunk('data: {"type":"content_block_start","index":0}', ctx()) is None
    assert p.parse_chunk('data: {"type":"message_start"}', ctx()) is None
    assert p.parse_chunk('data: {"type":"ping"}', ctx()) is None
    assert p.parse_chunk("event: content_block_delta", ctx()) is None


def test_parse_chunk_message_delta_carries_usage_and_stop():
    p = anthropic()
    chunk = p.parse_chunk(
        'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},'
        '"usage":{"input_tokens":4,"output_tokens":9}}',
        ctx(),
    )
    assert chunk["choices"][0]["finish_reason"] == "length"
    assert chunk["usage"]["total_tokens"] == 13


def test_parse_chunk_done_sentinel():
    p = anthropic()
    assert p.parse_chunk("data: [DONE]", ctx()) is DONE


def test_list_models_uses_transport():
    p = anthropic()
    transport = FakeTransport(
        responses=[UpstreamResponse(200, {"data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-opus-4"}]})]
    )

    async def go():
        return await p.list_models(transport)

    import asyncio

    assert asyncio.run(go()) == ["claude-sonnet-4-5", "claude-opus-4"]
    assert transport.requested[0].url.endswith("/models")
    assert transport.requested[0].headers["x-api-key"] == "sk-ant-test"


def test_completions_not_supported():
    p = anthropic()
    with pytest.raises(ApiError) as e:
        p.prepare_completions(ctx({"prompt": "p"}))
    assert e.value.status == 404