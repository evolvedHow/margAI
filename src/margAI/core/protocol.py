"""Transport contract.

Everything upstream is reached through a :class:`Transport`. Providers build
a :class:`PreparedRequest`; the transport executes it and returns a
:class:`UpstreamResponse` (one-shot) or an :class:`UpstreamStream` (SSE).
The core never performs I/O itself, so any runtime -- httpx on ASGI, fetch
on a Cloudflare Worker, a test double -- can supply a transport.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["PreparedRequest", "Transport", "UpstreamResponse", "UpstreamStream"]


@dataclass(slots=True)
class PreparedRequest:
    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    json: Any = None
    timeout: float | None = None  # per-request override; None -> transport default
    data: dict[str, str] | None = None  # multipart form fields; mutually exclusive with json
    files: Any = None  # multipart file parts, e.g. {"file": (name, bytes, type)}


@dataclass(slots=True)
class UpstreamResponse:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


class UpstreamStream(Protocol):
    """A held-open upstream response whose body has not been consumed."""

    status: int

    def lines(self) -> AsyncIterator[str]:
        """Iterate raw SSE lines (each line a single ``data: ...`` string)."""
        ...

    async def json(self) -> Any:
        """Read the full (already-buffered) body as JSON. Only valid before
        :meth:`lines` has been consumed and normally only for error bodies."""
        ...


@runtime_checkable
class Transport(Protocol):
    async def request(self, req: PreparedRequest) -> UpstreamResponse: ...

    async def open_stream(self, req: PreparedRequest) -> UpstreamStream: ...