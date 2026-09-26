"""Streaming pipeline tests."""

from __future__ import annotations

import json

import pytest

from margAI import ApiError, Telemetry
from margAI.config import TelemetryConfig
from margAI.core.errors import DONE
from margAI.providers.openai_compat import OpenAICompatProvider

from conftest import FakeStream, FakeTransport, chunk_payload, completion_chunk_payload, make_wrapper


async def collect(handle):
    return [line async for line in handle.lines()]


def sse_json_bytes(lines):
    return [
        json.loads(l[len("data: ") :])
        for l in lines
        if l.startswith("data: ") and not l.startswith("data: [DONE]")
    ]


def test_stream_hook_rewrites_and_forwards_chunks():
    raw = [
        f"data: {json.dumps(chunk_payload('He'))}",
        f"data: {json.dumps(chunk_payload('llo', finish_reason='stop'))}",
        "data: [DONE]",
    ]
    wrapper = make_wrapper(FakeTransport(streams=[FakeStream(raw)]))

    @wrapper.stream
    def shouting(chunk, ctx):
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            chunk["choices"][0]["delta"]["content"] = content.upper()
        return chunk

    import asyncio

    handle = asyncio.run(wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": [], "stream": True}))
    lines = asyncio.run(collect(handle))
    chunks = sse_json_bytes(lines)
    assert chunks[0]["choices"][0]["delta"]["content"] == "HE"
    assert chunks[1]["choices"][0]["delta"]["content"] == "LLO"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_can_drop_a_chunk():
    raw = [
        f"data: {json.dumps(chunk_payload('a'))}",
        f"data: {json.dumps(chunk_payload('b'))}",
        "data: [DONE]",
    ]
    wrapper = make_wrapper(FakeTransport(streams=[FakeStream(raw)]))

    @wrapper.stream
    def drop_b(chunk, ctx):
        if chunk["choices"][0]["delta"].get("content") == "b":
            return None
        return chunk

    lines = [line async for line in (await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": []})).lines()]
    contents = [
        json.loads(l[len("data: ") :])["choices"][0]["delta"]["content"]
        for l in lines
        if l.startswith("data: ") and not l.startswith("data: [DONE]")
    ]
    assert contents == ["a"]


async def test_open_stream_raises_on_upstream_error_status():
    wrapper = make_wrapper(
        FakeTransport(
            streams=[FakeStream([], status=429, error_json={"error": {"message": "quota", "type": "insufficient_quota", "code": "quota"}})]
        )
    )
    with pytest.raises(ApiError) as e:
        await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": []})
    assert e.value.status == 429
    assert e.value.message == "quota"


async def test_stream_error_hook_shapes_midstream_error():
    class ExplodingStream(FakeStream):
        def lines(self):
            async def _gen():
                yield f"data: {json.dumps(chunk_payload('almost done'))}"
                raise ApiError(502, "upstream dropped us", error_type="upstream_error")

            return _gen()

    wrapper = make_wrapper(FakeTransport(streams=[ExplodingStream([], )]))

    @wrapper.error
    def on_error(exc, ctx):
        return {"error": {"message": "connection interrupted", "type": "upstream_error", "param": None, "code": None}}

    lines = [line async for line in (await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": []})).lines()]
    error_chunk = json.loads(lines[1][6:-2])
    assert error_chunk["error"]["message"] == "connection interrupted"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_emits_telemetry_with_usage_from_last_chunk():
    records = []
    from margAI import Telemetry
    from margAI.config import TelemetryConfig

    tele = Telemetry(TelemetryConfig(enabled=True, emit="callback", callback="x"), callback=records.append)
    last = chunk_payload("bye", finish_reason="stop")
    last["usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    wrapper = make_wrapper(
        FakeTransport(streams=[FakeStream([f"data: {json.dumps(last)}", "data: [DONE]"])]), telemetry=tele
    )
    await collect(await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": []}))
    assert records, "a stream that ends must emit a telemetry record"
    assert records[0].prompt_tokens == 5
    assert records[0].stream is True


async def test_completions_stream_forwards_text_deltas():
    raw = [
        f"data: {json.dumps(completion_chunk_payload('ap'))}",
        f"data: {json.dumps(completion_chunk_payload('ple', finish_reason='stop'))}",
        "data: [DONE]",
    ]
    wrapper = make_wrapper(FakeTransport(streams=[FakeStream(raw)]))
    handle = await wrapper.open_stream(
        {"model": "margAI/openai/gpt-4o", "prompt": "p", "stream": True}, kind="completions"
    )
    lines = await collect(handle)
    texts = [
        json.loads(l[len("data: ") :])["choices"][0]["text"]
        for l in lines
        if l.startswith("data: ") and not l.startswith("data: [DONE]")
    ]
    assert texts == ["ap", "ple"]
    assert lines[-1] == "data: [DONE]\n\n"
    assert lines[0].startswith("data: {")  # SSE data frame preserved


def test_parse_chunk_handles_framing(provider_instance):
    assert provider_instance.parse_chunk("data: [DONE]", None) is DONE
    assert provider_instance.parse_chunk(": ping", None) is None
    assert provider_instance.parse_chunk("event: foo", None) is None
    assert provider_instance.parse_chunk("data: [DONE]", None) is DONE
    chunk = provider_instance.parse_chunk('data: {"choices": []}', None)
    assert chunk == {"choices": []}


@pytest.fixture
def provider_instance():
    from conftest import provider_config

    return OpenAICompatProvider(provider_config())