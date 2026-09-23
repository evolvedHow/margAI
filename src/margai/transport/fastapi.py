"""FastAPI adapter: expose a :class:`Wrapper` behind an OpenAI-compatible API.

This is the transport LibreChat (and any OpenAI SDK) points at. The wrapper
can be built from config, or injected directly (tests, custom wiring).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import Config, load_config
from ..core.errors import ApiError
from ..wrapper import Wrapper

__all__ = ["build_app"]

ERROR_HEADERS = {"X-Accel-Buffering": "no"}


def _error_response(status: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def build_app(wrapper: Wrapper | None = None, config: Config | None = None) -> FastAPI:
    """Build a FastAPI application around a wrapper.

    Pass either an existing :class:`Wrapper` (tests / embedded use) or a
    :class:`Config` to construct one from configuration.
    """
    wrapper = wrapper or Wrapper.from_config(config or load_config())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await wrapper.aclose()

    app = FastAPI(title=f"{wrapper.name} gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    async def root():
        return {"name": wrapper.name, "status": "ok", "prefix": wrapper.router.prefix}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": await wrapper.models()}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error_response(400, "Request body must be valid JSON")
        if not isinstance(body, dict):
            return _error_response(400, "Request body must be a JSON object")

        if body.get("stream", False):
            try:
                handle = await wrapper.open_stream(body)
            except ApiError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status)
            except Exception as exc:  # pragma: no cover - defensive
                return _error_response(500, str(exc), "server_error")
            return StreamingResponse(
                handle.lines(), media_type="text/event-stream", headers=ERROR_HEADERS
            )

        response = await wrapper.complete(body)
        return JSONResponse(content=response.body, status_code=response.status)

    return app