"""FastAPI adapter end-to-end tests via the ASGI TestClient."""

from __future__ import annotations

import json

import pytest
from conftest import (
    CHAT_MODEL,
    SSE_DONE,
    FakeStream,
    FakeTransport,
    chat_body,
    chat_payload,
    chunk_payload,
    completion_chunk_payload,
    make_wrapper,
    sse_line,
    user,
)
from fastapi.testclient import TestClient

from margAI.core.protocol import UpstreamResponse
from margAI.transport.fastapi import build_app


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
                        sse_line(chunk_payload("hi")),
                        sse_line(chunk_payload("", finish_reason="stop")),
                        SSE_DONE,
                    ]
                )
            ]
        )
    )
    app = build_app(streaming_wrapper)
    with TestClient(app) as c, c.stream(
        "POST", "/v1/chat/completions", json=chat_body(stream=True)
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
        contents = [
            json.loads(ln[len("data: ") :])["choices"][0]["delta"].get("content")
            for ln in lines
            if ln.startswith("data: ") and not ln.startswith("data: [DONE]")
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
            "/v1/chat/completions",
            json=chat_body(user("x")),
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
                UpstreamResponse(
                    200,
                    {
                        "object": "text_completion",
                        "choices": [{"index": 0, "text": "pong", "finish_reason": "stop"}],
                    },
                )
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
                        sse_line(completion_chunk_payload("ap")),
                        sse_line(completion_chunk_payload("ple", finish_reason="stop")),
                        SSE_DONE,
                    ]
                )
            ]
        )
    )
    app = build_app(wrapper)
    with TestClient(app) as c, c.stream(
        "POST", "/v1/completions", json={"model": CHAT_MODEL, "prompt": "p", "stream": True}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
        texts = [
            json.loads(ln[len("data: ") :])["choices"][0]["text"]
            for ln in lines
            if ln.startswith("data: ") and not ln.startswith("data: [DONE]")
        ]
        assert texts == ["ap", "ple"]
        assert lines[-1] == "data: [DONE]"


def test_completions_invalid_json_returns_400(client):
    resp = client.post("/v1/completions", content="nope", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# -- OpenAPI documentation ----------------------------------------------------
#
# The generated document is the API reference, so these assert on the
# *document*, not on runtime behaviour: a field or a status that loses its
# description is a documentation bug even though every call still succeeds.

#: Every routed endpoint, and the OpenAPI tag it is filed under.
DOCUMENTED_ROUTES = {
    "/v1/models": ("get", "Models"),
    "/v1/models/{model_id}": ("get", "Models"),
    "/v1/chat/completions": ("post", "Chat"),
    "/v1/completions": ("post", "Completions"),
    "/v1/embeddings": ("post", "Embeddings"),
    "/v1/images/generations": ("post", "Images"),
    "/v1/audio/transcriptions": ("post", "Audio"),
}


@pytest.fixture(scope="module")
def openapi():
    return build_app().openapi()


def test_every_route_is_in_the_document_with_a_summary_and_description(openapi):
    for path, (method, tag) in DOCUMENTED_ROUTES.items():
        operation = openapi["paths"][path][method]
        assert operation["tags"] == [tag], f"{path} is filed under the wrong tag"
        assert operation["summary"], f"{path} has no summary"
        assert operation["description"], f"{path} has no description"


def test_the_tag_introspection_endpoint_is_documented(openapi):
    operation = openapi["paths"]["/v1/margAI/tags"]["get"]
    assert operation["summary"]
    assert operation["description"]


def test_the_documented_tags_schema_covers_what_the_endpoint_returns():
    """`wrapper.describe()` is the payload. A key it returns but the docs do
    not name is a field an operator discovers by reading `describe()`'s source
    rather than by reading `/docs`, which is the reason this endpoint exists."""
    from margAI.config import Config, GatewayConfig

    with TestClient(build_app(config=Config(gateway=GatewayConfig()))) as client:
        body = client.get("/v1/margAI/tags").json()

    schema = build_app().openapi()["components"]["schemas"]["GatewayDescriptionResponse"]
    undocumented = sorted(set(body) - set(schema["properties"]))
    assert undocumented == []


def test_no_request_field_is_undocumented(openapi):
    """A parameter whose meaning is not written down is a parameter the docs
    get wrong, so every one of them carries a description."""
    undocumented = []
    for path, (method, _tag) in DOCUMENTED_ROUTES.items():
        operation = openapi["paths"][path][method]
        body = (operation.get("requestBody") or {}).get("content", {})
        for schema_ref in (v.get("schema", {}) for v in body.values()):
            if "$ref" not in schema_ref:
                continue
            name = schema_ref["$ref"].rsplit("/", 1)[-1]
            schema = openapi["components"]["schemas"][name]
            for field, prop in (schema.get("properties") or {}).items():
                if not prop.get("description"):
                    undocumented.append(f"{path} -> {name}.{field}")
    assert undocumented == []


def test_every_error_status_is_documented(openapi):
    for path, (method, _tag) in DOCUMENTED_ROUTES.items():
        responses = openapi["paths"][path][method]["responses"]
        for status in ("400", "404", "500"):
            assert status in responses, f"{path} does not document {status}"
            assert responses[status]["description"], f"{path} {status} has no description"


def test_a_422_is_documented_as_never_returned(openapi):
    """margAI replaces FastAPI's validation handler, so a validation failure is
    a 400. The generated document must say so rather than advertising a status
    the server never returns."""
    responses = openapi["paths"]["/v1/chat/completions"]["post"]["responses"]
    assert "400" in responses
    assert "Never returned" in responses["422"]["description"]


def test_chat_documents_both_json_and_streaming_shapes(openapi):
    content = openapi["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"]
    assert "application/json" in content
    assert "text/event-stream" in content


def test_transcription_documents_every_response_format(openapi):
    content = openapi["paths"]["/v1/audio/transcriptions"]["post"]["responses"]["200"]["content"]
    assert "application/json" in content
    assert "text/plain" in content
    assert "application/x-subrip" in content
    assert "text/vtt" in content


def test_the_app_description_names_the_configured_namespace():
    """The namespace is configurable, so the document must not hardcode one."""
    schema = build_app().openapi()
    assert "margAI" in schema["info"]["description"]
    assert "<namespace>" not in schema["info"]["description"]
