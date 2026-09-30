"""Pipeline control from a handler: route, answer, or fail the call.

`ctx.route_to()`, `ctx.respond()` and `ctx.error()` are how a handler takes over
a call. These tests exist because those three return values used to be discarded
-- a handler could short-circuit a call and the gateway would serve it from the
model anyway, which is the worst possible failure for a cache: it works, and it
is silently doing the wrong thing.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeTransport, UpstreamResponse, chat_payload, make_wrapper

from margAI.core.errors import ApiError
from margAI.telemetry import Telemetry, TelemetryConfig


def recording_telemetry(rows: list) -> Telemetry:
    return Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=rows.append)


def tagged(tag: str, *, model: str = "margAI/openai/gpt-4o") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"!margAI: {tag}\nhello"}],
    }


class Exploding(FakeTransport):
    """A transport that fails the test if it is ever called."""

    async def request(self, prepared):
        raise AssertionError("the provider must not be called")


# -- route_to -----------------------------------------------------------------


async def test_route_to_pins_the_model_the_provider_is_asked_for():
    """The override has to reach the wire, not just the intent."""
    seen: dict = {}

    class Recording(FakeTransport):
        async def request(self, prepared):
            seen["model"] = (prepared.json or {}).get("model")
            return UpstreamResponse(200, chat_payload("ok"))

    app = make_wrapper(Recording())
    app.on("pin", phase="before")(lambda ctx, tag: ctx.route_to("margAI/openai/gpt-4o-mini"))
    await app.complete(tagged("pin"))
    assert seen["model"] == "gpt-4o-mini"


async def test_route_to_records_why_the_route_was_chosen():
    """Telemetry attributes every dynamic call; a handler-pinned one has to be
    attributable too, or the override is invisible in a report."""
    rows: list = []
    app = make_wrapper(FakeTransport(), telemetry=recording_telemetry(rows))
    app.on("pin", phase="before")(lambda ctx, tag: ctx.route_to("margAI/openai/gpt-4o-mini"))
    await app.complete(tagged("pin"))
    assert rows[0].route_reason == "handler:route"


async def test_route_to_does_not_rewrite_what_the_caller_asked_for():
    """`request_model` is what the client sent. Losing it would make the
    override invisible in a cost report, which is exactly where you look when a
    model swap is suspected."""
    rows: list = []
    app = make_wrapper(FakeTransport(), telemetry=recording_telemetry(rows))
    app.on("pin", phase="before")(lambda ctx, tag: ctx.route_to("margAI/openai/gpt-4o-mini"))
    await app.complete(tagged("pin"))
    assert rows[0].request_model == "margAI/openai/gpt-4o"
    assert rows[0].upstream_model == "gpt-4o-mini"


async def test_the_handlers_after_a_route_override_do_not_run():
    """A signal is a decision about the whole call. Running the handlers behind
    it would act on a route that is no longer the one being taken."""

    def must_not_run(ctx, tag):
        raise AssertionError("ran after a signal")

    app = make_wrapper(FakeTransport())
    app.on("pin", phase="before")(lambda ctx, tag: ctx.route_to("margAI/openai/gpt-4o-mini"))
    app.on("other", phase="before")(must_not_run)
    assert (await app.complete(tagged("pin other"))).status == 200


# -- respond ------------------------------------------------------------------


async def test_respond_answers_without_calling_the_provider():
    """The point of a cache: the model is never asked."""
    cached = {"id": "chatcmpl-cached", "object": "chat.completion", "choices": []}
    app = make_wrapper(Exploding())
    app.on("cache", phase="before")(lambda ctx, tag: ctx.respond(cached))
    response = await app.complete(tagged("cache"))
    assert response.status == 200
    assert response.body == cached


async def test_respond_works_for_a_model_that_does_not_exist():
    """A short-circuit happens before the provider is resolved, so an answer
    that has nothing to do with the requested model still succeeds. A pinned
    provider, or a stale model id in a client's config, is the common case."""
    app = make_wrapper(Exploding())
    app.on("cache", phase="before")(lambda ctx, tag: ctx.respond({"choices": []}))
    response = await app.complete(tagged("cache", model="provider/does-not-exist"))
    assert response.status == 200


async def test_respond_is_recorded_so_a_hit_rate_is_measurable():
    rows: list = []
    app = make_wrapper(FakeTransport(), telemetry=recording_telemetry(rows))
    app.on("cache", phase="before")(lambda ctx, tag: ctx.respond({"choices": []}))
    await app.complete(tagged("cache"))
    assert rows[0].route_reason == "handler:answer"
    assert rows[0].status == 200


async def test_respond_over_a_stream_answers_with_one_frame_and_the_terminator():
    """The client asked for SSE. A JSON body would leave it parsing a stream
    that never arrives, so the cached body goes out as a single frame."""
    cached = {"id": "chatcmpl-cached", "object": "chat.completion", "choices": []}
    app = make_wrapper(Exploding())
    app.on("cache", phase="before")(lambda ctx, tag: ctx.respond(cached))
    handle = await app.open_stream({**tagged("cache"), "stream": True})
    lines = [line async for line in handle.lines()]
    assert lines[-1] == "data: [DONE]\n\n"
    assert json.loads(lines[0][len("data: ") :]) == cached


# -- error --------------------------------------------------------------------


async def test_error_fails_the_call_with_the_given_status_and_type():
    app = make_wrapper(FakeTransport())
    app.on("validate", phase="before")(lambda ctx, tag: ctx.error(429, "slow down", "rate_limit_error"))
    response = await app.complete(tagged("validate"))
    assert response.status == 429
    assert response.body == {
        "error": {"message": "slow down", "type": "rate_limit_error", "param": None, "code": None}
    }


async def test_error_before_any_provider_work():
    """A rejected call should not cost a provider round trip."""
    app = make_wrapper(Exploding())
    app.on("validate", phase="before")(lambda ctx, tag: ctx.error(400, "too long"))
    assert (await app.complete(tagged("validate"))).status == 400


async def test_error_over_a_stream_still_answers_json_before_any_sse_byte():
    """`open_stream` raises `ApiError` in preflight so the transport can send a
    JSON error. A streaming client gets the same error it would get from any
    other rejection, rather than a stream-shaped one it cannot read."""
    app = make_wrapper(Exploding())
    app.on("validate", phase="before")(lambda ctx, tag: ctx.error(429, "slow down"))
    with pytest.raises(ApiError) as caught:
        await app.open_stream({**tagged("validate"), "stream": True})
    assert caught.value.status == 429
    assert caught.value.body["error"]["message"] == "slow down"


async def test_an_unresolvable_model_is_reached_when_nothing_answers():
    """The short-circuit is a handler's choice, not a way to make every call
    succeed: with no handler, the model still has to exist."""
    app = make_wrapper(FakeTransport())
    assert (await app.complete(tagged("nothing", model="provider/does-not-exist"))).status == 404
