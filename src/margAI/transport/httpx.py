"""httpx-based transport for the ASGI/uvicorn runtime (also runs on Modal
and anywhere else a plain Python process can use httpx).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..core.errors import ApiError
from ..core.protocol import PreparedRequest, UpstreamResponse

__all__ = ["HttpxStream", "HttpxTransport"]


class HttpxStream:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code

    async def json(self) -> Any:
        if not self._response.is_closed:
            await self._response.aread()
        return self._response.json()

    def lines(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            try:
                async for line in self._response.aiter_lines():
                    yield line
            finally:
                if not self._response.is_closed:
                    await self._response.aclose()

        return _gen()


class HttpxTransport:
    """Minimal :class:`Transport` backed by an ``httpx.AsyncClient``.

    Used by the FastAPI adapter and importable standalone. Create one client
    and reuse it (per-request clients leak connections).
    """

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float | None = None) -> None:
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(timeout if timeout is not None else 60.0))
        self._client = client

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build(self, req: PreparedRequest) -> httpx.Request:
        if (req.data or req.files) and req.json is not None:
            raise ApiError(
                500, "PreparedRequest cannot carry both a JSON body and multipart data", error_type="server_error"
            )
        return self._client.build_request(
            method=req.method,
            url=req.url,
            headers=req.headers,
            json=req.json,
            data=req.data,
            files=req.files,
            timeout=req.timeout,
        )

    async def request(self, req: PreparedRequest) -> UpstreamResponse:
        try:
            response = await self._client.send(self._build(req))
        except httpx.TimeoutException as exc:
            raise ApiError(
                504, f"Upstream request timed out ({req.url})", error_type="server_error", code="timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                502, f"Upstream request failed ({req.url}): {exc}", error_type="server_error"
            ) from exc
        body: Any = None
        if response.content:
            try:
                body = response.json()
            except ValueError:
                body = response.text
        return UpstreamResponse(status=response.status_code, body=body, headers=dict(response.headers))

    async def open_stream(self, req: PreparedRequest) -> HttpxStream:
        try:
            response = await self._client.send(self._build(req), stream=True)
        except httpx.TimeoutException as exc:
            raise ApiError(
                504, f"Upstream request timed out ({req.url})", error_type="server_error", code="timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                502, f"Upstream request failed ({req.url}): {exc}", error_type="server_error"
            ) from exc
        return HttpxStream(response)