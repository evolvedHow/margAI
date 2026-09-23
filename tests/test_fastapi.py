"""FastAPI adapter end-to-end tests via the ASGI TestClient."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from margai import ApiError
from margai.core.protocol import UpstreamResponse
from margai.transport.fastapi import build_app
from margai.wrapper import GatewayResponse

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
    assert "marg/openai/gpt-4o" in ids
    assert all(m["object"] == "model" for m in data)


def test_non_stream_chat(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "marg/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
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
        with c.stream("POST", "/v1/chat/completions", json={"model": "marg/openai/gpt-4o", "messages": [], "stream": True}) as r:
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


def test_unprefixed_model_404(client):
    resp = client.post(
        "/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body


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
            "/v1/chat/completions", json={"model": "marg/openai/gpt-4o", "messages": [{"role": "user", "content": "x"}]}
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["message"] == "rl"


def test_root_fastapi_app_info(client):
    resp = client.get("/")
    assert resp.json()["status"] == "ok"
    assert resp.json()["prefix"] == "marg"