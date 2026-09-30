"""Streaming pipeline tests."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import (
    FakeStream,
    FakeTransport,
    chat_body,
    chunk_payload,
    completion_chunk_payload,
    make_wrapper,
)

from margAI import ApiError
from margAI.config import TelemetryConfig
from margAI.core.errors import DONE
from margAI.providers.openai_compat import OpenAICompatProvider
from margAI.telemetry import Telemetry


async def collect(handle):
    return [line async for line in handle.lines()]


def sse_json_bytes(lines):
    return [
        json.loads(ln[len("data: ") :])
        for ln in lines
        if ln.startswith("data: ") and not ln.startswith("data: [DONE]")
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

    lines = [line async for line in (await wrapper.open_stream(chat_body())).lines()]
    contents = [
        json.loads(ln[len("data: ") :])["choices"][0]["delta"]["content"]
        for ln in lines
        if ln.startswith("data: ") and not ln.startswith("data: [DONE]")
    ]
    assert contents == ["a"]


async def test_open_stream_raises_on_upstream_error_status():
    wrapper = make_wrapper(
        FakeTransport(
            streams=[
                FakeStream(
                    [],
                    status=429,
                    error_json={
                        "error": {
                            "message": "quota",
                            "type": "insufficient_quota",
                            "code": "quota",
                        }
                    },
                )
            ]
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

    lines = [line async for line in (await wrapper.open_stream(chat_body())).lines()]
    error_chunk = json.loads(lines[1][6:-2])
    assert error_chunk["error"]["message"] == "connection interrupted"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_stream_emits_telemetry_with_usage_from_last_chunk():
    records: list = []
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
        json.loads(ln[len("data: ") :])["choices"][0]["text"]
        for ln in lines
        if ln.startswith("data: ") and not ln.startswith("data: [DONE]")
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

# -- preflight failures ---------------------------------------------------


def test_a_non_streamable_kind_is_reported_not_swallowed():
    """The guard used to sit outside the try, so an embeddings `stream=True`
    raised a bare ApiError: no error hook, no telemetry, and a caller had no
    way to tell the call never happened."""
    w = make_wrapper(FakeTransport())
    with pytest.raises(ApiError) as exc:
        asyncio.run(w.open_stream(chat_body(), kind="embeddings"))
    assert exc.value.status == 400
    assert "does not support streaming" in exc.value.message


def test_a_non_streamable_kind_is_still_recorded_in_telemetry():
    """Moving the guard inside the try is what buys this. It cannot reach the
    error hooks -- those are `fn(exc, ctx)` and no context exists yet -- but a
    rejected call must not vanish from the record either."""
    records: list = []
    w = make_wrapper(
        FakeTransport(),
        telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append),
    )
    with pytest.raises(ApiError):
        asyncio.run(w.open_stream(chat_body(), kind="embeddings"))
    assert len(records) == 1
    assert records[0].status == 400
    assert records[0].error == "Kind 'embeddings' does not support streaming"


def test_an_error_hook_body_survives_a_non_ApiError_preflight_failure():
    """A non-ApiError failure used to drop whatever the error hook said, so
    the hook only worked for the one exception class that already had a
    body -- which is the opposite of when you need it."""
    class Boom(Exception):
        pass

    async def explode(req):
        raise Boom("upstream on fire")

    w = make_wrapper(FakeTransport(streams=[explode]))

    @w.error
    def shape(exc, ctx):
        return {"error": {"message": f"shaped: {type(exc).__name__}", "type": "shaped"}}

    with pytest.raises(ApiError) as exc:
        asyncio.run(w.open_stream(chat_body()))
    assert exc.value.status == 500
    assert exc.value.body == {"error": {"message": "shaped: Boom", "type": "shaped"}}


def test_a_non_ApiError_preflight_failure_without_a_hook_body_reraises_the_original():
    """Nothing shaped the body, so the original exception propagates rather
    than being wrapped. `transport.fastapi` catches it and answers a redacted
    500; the traceback stays intact for the log."""
    def explode(req):
        raise RuntimeError("nope")

    w = make_wrapper(FakeTransport(streams=[explode]))
    with pytest.raises(RuntimeError, match="nope"):
        asyncio.run(w.open_stream(chat_body()))
