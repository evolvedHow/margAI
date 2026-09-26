"""Provider interface and OpenAI-compatible provider.

A provider owns three jobs:

  1. ``prepare_*`` -- turn the internal (OpenAI-shaped) body into an upstream
     :class:`PreparedRequest` (URL, headers, native payload).
  2. ``parse_response`` / ``parse_chunk`` -- turn native responses back into
     the OpenAI shape the framework (and its hooks) operate on.
  3. ``list_models`` -- return upstream model ids for the catalog.

The supported surface is deliberately small -- chat, legacy completions,
embeddings, image generation, and audio transcription. Everything here is
exercised end-to-end by the test suite; anything not modelled here is not
routed. See ``docs/ENDPOINTS.md`` for the reasoning and for how to extend
the surface with a declarative endpoint table.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..config import ProviderConfig
from ..core import DONE
from ..core.errors import ApiError
from ..core.protocol import PreparedRequest, Transport, UpstreamResponse

__all__ = ["OpenAICompatProvider", "Provider"]


class Provider(ABC):
    """Base class for upstream providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    def configured_models(self) -> list[str]:
        """Model ids declared in config. Used by the router (offline)."""
        return list(self.config.models)

    def headers(self, *, json_body: bool = True) -> dict[str, str]:
        """Upstream headers. ``json_body=False`` for multipart requests, where
        the transport must set ``Content-Type`` (and its boundary) itself."""
        headers = {"Content-Type": "application/json"} if json_body else {}
        extra = self.config.extra
        if isinstance(extra, dict) and isinstance(extra.get("headers"), dict):
            headers.update({str(k): str(v) for k, v in extra["headers"].items()})
        key = self.config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def endpoint(self, *parts: str) -> str:
        base = self.config.base_url.rstrip("/")
        return "/".join([base, *parts])

    # -- chat ---------------------------------------------------------------

    @abstractmethod
    def prepare_chat(self, ctx: Any) -> PreparedRequest:
        """Build the upstream request from a prepared :class:`RequestContext`."""

    @abstractmethod
    def parse_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        """Map a non-streaming upstream response to ``(openai_shaped_dict, status)``.

        Raise :class:`ApiError` for upstream error statuses.
        """

    @abstractmethod
    def parse_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        """Map one raw SSE line to an OpenAI chunk dict.

        Return ``None`` to skip the line, or :data:`margAI.core.DONE` to end
        the stream.
        """

    # -- legacy text completions ----------------------------------------------
    # Providers that don't speak the legacy completions surface (Anthropic, for
    # one) inherit these 404-raising fallbacks, so `/v1/completions` degrades
    # cleanly instead of 500-ing.

    def prepare_completions(self, ctx: Any) -> PreparedRequest:
        raise _unsupported(self.name, "/v1/completions")

    def parse_completion_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise _unsupported(self.name, "/v1/completions")

    def parse_completion_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        raise _unsupported(self.name, "/v1/completions")

    def extract_usage(self, payload: dict) -> dict[str, Any]:
        """Pull token usage out of an OpenAI-shaped payload/chunk."""
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return usage if isinstance(usage, dict) else {}

    # -- catalog ------------------------------------------------------------

    @abstractmethod
    async def list_models(self, transport: Transport) -> list[str]:
        """Fetch model ids from the upstream (falls back to configured list)."""


class OpenAICompatProvider(Provider):
    """Anything that speaks the OpenAI HTTP surface (OpenAI, OpenRouter,
    vLLM, Ollama, LM Studio, ...)."""

    # -- chat ---------------------------------------------------------------

    def prepare_chat(self, ctx: Any) -> PreparedRequest:
        return self._json(ctx.body, "chat", "completions")

    def parse_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        return _openai_body(resp, {"choices": []})

    def parse_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        return _parse_sse_data_line(raw_line)

    # -- legacy completions --------------------------------------------------

    def prepare_completions(self, ctx: Any) -> PreparedRequest:
        return self._json(ctx.body, "completions")

    def parse_completion_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        return _openai_body(resp, {"choices": []})

    def parse_completion_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        return _parse_sse_data_line(raw_line)

    # -- embeddings ----------------------------------------------------------

    def prepare_embeddings(self, ctx: Any) -> PreparedRequest:
        return self._json(ctx.body, "embeddings")

    def parse_embeddings_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        return _openai_body(resp, {"data": []})

    # -- images --------------------------------------------------------------

    def prepare_images_generations(self, ctx: Any) -> PreparedRequest:
        return self._json(ctx.body, "images", "generations")

    def parse_images_generations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        return _openai_body(resp, {"data": []})

    # -- audio ---------------------------------------------------------------

    def prepare_audio_transcriptions(self, ctx: Any) -> PreparedRequest:
        body = dict(ctx.body or {})
        upload = body.pop("file", None)
        if upload is None:
            raise ApiError(
                400,
                "Audio transcription requires a 'file' upload (multipart/form-data).",
                error_type="invalid_request_error",
                param="file",
            )
        fields = {k: _form_value(v) for k, v in body.items() if v is not None}
        return PreparedRequest(
            method="POST",
            url=self.endpoint("audio", "transcriptions"),
            headers=self.headers(json_body=False),
            data=fields,
            files={"file": upload},
            timeout=self.config.timeout,
        )

    def parse_audio_transcriptions_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        return _openai_body(resp, {"text": ""})

    # -- catalog -------------------------------------------------------------

    async def list_models(self, transport: Transport) -> list[str]:
        resp = await transport.request(
            PreparedRequest(
                method="GET",
                url=self.endpoint("models"),
                headers=self.headers(),
                timeout=self.config.timeout,
            )
        )
        if resp.status >= 400:
            raise _error_from_response(resp, resp.status)
        data = resp.body.get("data", []) if isinstance(resp.body, dict) else []
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]

    # -- helpers -------------------------------------------------------------

    def _json(self, body: Any, *parts: str) -> PreparedRequest:
        """A JSON passthrough: the internal body is already OpenAI-shaped and
        ``model`` has already been rewritten by the router."""
        return PreparedRequest(
            method="POST",
            url=self.endpoint(*parts),
            headers=self.headers(),
            json=body,
            timeout=self.config.timeout,
        )


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _openai_body(resp: UpstreamResponse, empty: dict) -> tuple[dict, int]:
    status = resp.status
    if status >= 400:
        raise _error_from_response(resp, status)
    return (resp.body if isinstance(resp.body, dict) else dict(empty)), status


def _unsupported(provider: str, endpoint: str) -> ApiError:
    return ApiError(
        404,
        f"Provider '{provider}' does not support {endpoint}",
        error_type="invalid_request_error",
        param="model",
    )


def _parse_sse_data_line(raw_line: str) -> dict | None:
    line = raw_line.strip()
    if not line or line.startswith(":"):
        return None  # keepalive or comment
    if not line.startswith("data:"):
        return None  # unknown framing, skip
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return DONE
    try:
        return json.loads(data)
    except ValueError:
        return None


def _error_from_response(resp: UpstreamResponse, status: int) -> ApiError:
    return ApiError.from_openai_body(resp.body, implicit_status=status)
