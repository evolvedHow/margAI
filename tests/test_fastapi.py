"""FastAPI adapter end-to-end tests via the ASGI TestClient."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from margAI import ApiError
from margAI.core.protocol import UpstreamResponse
from margAI.transport.fastapi import build_app
from margAI.wrapper import GatewayResponse

from conftest import (
    FakeStream,
    FakeTransport,
    chat_payload,
    chunk_payload,
    make_wrapper,
)


@pytest.fixture
def client():
    wrapper = make_wrapper(
        FakeTransport(
            responses=[
                UpstreamResponse(200, chat_payload("pong")),
                UpstreamResponse(200, chat_payload("pong")),
                UpstreamResponse(200, chat_payload("pong")),
            ]
        )
    )
    app = build_app(wrapper)
    with TestClient(app) as c:
        yield c


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()["data"]
    ids = [m["id"] for m in data]
    assert "margAI/openai/gpt-4o" in ids
    assert all(m["object"] == "model" for m in data)


def test_non_stream_chat(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "pong"


def test_stream_chat(client):
    streaming_wrapper = make_wrapper(
        FakeTransport(
            streams=[
                FakeStream(
                    [
                        'data: {"id": "cmpl", "object": "chat.completion.chunk", "model": "echo", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]}',
                        'data: {"id": "cmpl", "object": "chat.completion.chunk", "model": "echo", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
                        "data: [DONE]",
                    ]
                )
            ]
        )
    )
    app = build_app(streaming_wrapper)
    with TestClient(app) as c:
        with c.stream("POST", "/v1/chat/completions", json={"model": "margAI/openai/gpt-4o", "messages": [], "stream": True}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = [l for l in r.iter_lines() if l]
            contents = [
                json.loads(l[len("data: ") :])["choices"][0]["delta"].get("content")
                for l in lines
                if l.startswith("data: ") and not l.startswith("data: [DONE]")
            ]
            assert contents[0] == "hi"
            assert lines[-1] == "data: [DONE]"


def test_invalid_json_returns_400(client):
    resp = client.post("/v1/chat/completions", content="not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_non_dict_body_returns_400(client):
    resp = client.post("/v1/chat/completions", content="[1,2]", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_bare_model_id_is_accepted_despite_prefixed_expose(client):
    """`expose` controls what /v1/models *lists*, not what the router accepts.

    Conflating the two meant `expose = "prefixed"` silently rejected every bare
    model id even with a default_provider set, and told the operator nowhere.
    """
    resp = client.post(
        "/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200


def test_unknown_model_still_404s(client):
    resp = client.post(
        "/v1/chat/completions", json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_upstream_error_propagates_status(client):
    err_wrapper = make_wrapper(
        FakeTransport(
            responses=[
                UpstreamResponse(429, {"error": {"message": "rl", "type": "rate_limit_error"}})
            ]
        )
    )
    app = build_app(err_wrapper)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/chat/completions", json={"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "x"}]}
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["message"] == "rl"


def test_root_fastapi_app_info(client):
    resp = client.get("/")
    assert resp.json()["status"] == "ok"
    assert resp.json()["prefix"] == "margAI"


def test_completions_endpoint():
    wrapper = make_wrapper(
        FakeTransport(
            responses=[
                UpstreamResponse(200, {"object": "text_completion", "choices": [{"index": 0, "text": "pong", "finish_reason": "stop"}]})
            ]
        )
    )
    app = build_app(wrapper)
    with TestClient(app) as c:
        resp = c.post("/v1/completions", json={"model": "margAI/openai/gpt-4o", "prompt": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "pong"


def test_completions_stream_endpoint():
    wrapper = make_wrapper(
        FakeTransport(
            streams=[
                FakeStream(
                    [
                        'data: {"object": "text_completion", "choices": [{"index": 0, "text": "ap", "finish_reason": null}]}',
                        'data: {"object": "text_completion", "choices": [{"index": 0, "text": "ple", "finish_reason": "stop"}]}',
                        "data: [DONE]",
                    ]
                )
            ]
        )
    )
    app = build_app(wrapper)
    with TestClient(app) as c:
        with c.stream("POST", "/v1/completions", json={"model": "margAI/openai/gpt-4o", "prompt": "p", "stream": True}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = [l for l in r.iter_lines() if l]
            texts = [
                json.loads(l[len("data: ") :])["choices"][0]["text"]
                for l in lines
                if l.startswith("data: ") and not l.startswith("data: [DONE]")
            ]
            assert texts == ["ap", "ple"]
            assert lines[-1] == "data: [DONE]"


def test_completions_invalid_json_returns_400(client):
    resp = client.post("/v1/completions", content="nope", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"