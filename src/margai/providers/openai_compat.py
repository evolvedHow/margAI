"""Provider interface and OpenAI-compatible provider.

A provider owns three jobs:
  1. ``prepare_chat`` -- turn the internal (OpenAI-shaped) body into an
     upstream :class:`PreparedRequest` (URL, headers, native payload).
  2. ``parse_response`` / ``parse_chunk`` -- turn native responses back into
     the OpenAI shape the framework (and its hooks) operate on.
  3. ``list_models`` -- return upstream model ids for the catalog.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..config import ProviderConfig
from ..core import DONE
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

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        extra = self.config.extra
        if isinstance(extra, dict) and isinstance(extra.get("headers"), dict):
            headers.update({str(k): str(v) for k, v in extra["headers"].items()})
        key = self.config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def timeout(self, default: float) -> float:
        return self.config.timeout or default

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

        Return ``None`` to skip the line, or :data:`margai.core.DONE` to end
        the stream.
        """

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

    def prepare_chat(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("chat", "completions"),
            headers=self.headers(),
            json=ctx.body,  # already OpenAI-shaped; `model` already rewritten
        )

    def parse_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"choices": []}), status

    def parse_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        return _parse_sse_data_line(raw_line)

    async def list_models(self, transport: Transport) -> list[str]:
        resp = await transport.request(
            PreparedRequest(method="GET", url=self.endpoint("models"), headers=self.headers())
        )
        if resp.status >= 400:
            raise _error_from_response(resp, resp.status)
        data = resp.body.get("data", []) if isinstance(resp.body, dict) else []
        ids = []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        return ids


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
        from json import loads

        return loads(data)
    except ValueError:
        return None


def _error_from_response(resp: UpstreamResponse, status: int):
    from ..core.errors import ApiError

    return ApiError.from_openai_body(resp.body, implicit_status=status)