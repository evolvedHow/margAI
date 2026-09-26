"""FastAPI adapter: expose a :class:`Wrapper` behind an OpenAI-compatible API.

This is the transport LibreChat (and any OpenAI SDK) points at. The wrapper
can be built from config, or injected directly (tests, custom wiring).

The routed surface is chat, legacy completions, embeddings, image
generation, and audio transcription -- see ``docs/ENDPOINTS.md``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Union

from fastapi import FastAPI, Request, File, UploadFile, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from ..config import Config, load_config
from ..core.errors import ApiError
from ..wrapper import Wrapper

__all__ = ["build_app"]

ERROR_HEADERS = {"X-Accel-Buffering": "no"}


# =============================================================================
# Pydantic request models
# =============================================================================

class ChatCompletionMessageParam(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "function", "tool"]
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None


class ChatCompletionToolParam(BaseModel):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessageParam]
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    response_format: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[dict[str, Any]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tools: Optional[list[ChatCompletionToolParam]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], dict[str, Any]]] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list[str], list[int], list[list[int]]]
    best_of: Optional[int] = Field(default=None, ge=1)
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, int]] = None
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[dict[str, Any]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str], list[int], list[list[int]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[Literal["standard", "hd"]] = "standard"
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] = "1024x1024"
    style: Optional[Literal["vivid", "natural"]] = "vivid"
    user: Optional[str] = None


class AudioTranscriptionRequest(BaseModel):
    """The non-file fields of a ``multipart/form-data`` transcription."""

    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = "json"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None


# =============================================================================
# Response models
# =============================================================================

class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelResponse]


# =============================================================================
# Error shaping
# =============================================================================

def _error_response(status: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def _validation_message(exc: RequestValidationError) -> str:
    """A useful message. The blanket "must be valid JSON" hides real schema
    errors (a missing field, a bad enum, an out-of-range number) behind a
    wrong diagnosis."""
    parts: list[str] = []
    for err in exc.errors():
        msg = err.get("msg", "invalid value")
        if "JSON decode error" in msg:
            return "Request body must be valid JSON"
        loc = ".".join(
            str(item) for item in err.get("loc", ()) if item not in ("body", "__root__") and not isinstance(item, int)
        )
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Request body must be valid JSON"


# =============================================================================
# FastAPI App Builder
# =============================================================================

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

    from .. import __version__

    app = FastAPI(
        title=f"{wrapper.name} gateway",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(400, _validation_message(exc), "invalid_request_error")

    # -------------------------------------------------------------------------
    # Root
    # -------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": wrapper.name, "status": "ok", "prefix": wrapper.router.prefix}

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    @app.get("/v1/models", response_model=ModelsResponse, tags=["Models"])
    async def list_models():
        return {"object": "list", "data": await wrapper.models()}

    @app.get("/v1/models/{model_id}", response_model=ModelResponse, tags=["Models"])
    async def retrieve_model(model_id: str):
        for model in await wrapper.models():
            if model.get("id") == model_id:
                return model
        return _error_response(404, f"Model '{model_id}' not found", "not_found_error")

    # -------------------------------------------------------------------------
    # Chat Completions
    # -------------------------------------------------------------------------
    @app.post("/v1/chat/completions", tags=["Chat"])
    async def chat_completions(body: ChatCompletionRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="chat")

    # -------------------------------------------------------------------------
    # Completions (Legacy)
    # -------------------------------------------------------------------------
    @app.post("/v1/completions", tags=["Completions"])
    async def completions(body: CompletionRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="completions")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    @app.post("/v1/embeddings", tags=["Embeddings"])
    async def embeddings(body: EmbeddingRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="embeddings")

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------
    @app.post("/v1/images/generations", tags=["Images"])
    async def images_generations(body: ImageGenerationRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="images_generations")

    # -------------------------------------------------------------------------
    # Audio (multipart: the only genuinely multipart endpoint we route)
    # -------------------------------------------------------------------------
    @app.post("/v1/audio/transcriptions", tags=["Audio"])
    async def audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Body(...),
        language: Optional[str] = Body(default=None),
        prompt: Optional[str] = Body(default=None),
        response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Body(default="json"),
        temperature: Optional[float] = Body(default=None, ge=0.0, le=1.0),
        timestamp_granularities: Optional[str] = Body(default=None),
    ):
        content = await file.read()
        if not content:
            return _error_response(400, "Uploaded audio file is empty", "invalid_request_error")
        body: dict[str, Any] = {
            "model": model,
            "file": (file.filename or "audio", content, file.content_type or "application/octet-stream"),
        }
        for key, value in (
            ("language", language),
            ("prompt", prompt),
            ("response_format", response_format),
            ("temperature", temperature),
            ("timestamp_granularities", timestamp_granularities),
        ):
            if value is not None:
                body[key] = value
        return await _forward(body, wrapper, kind="audio_transcriptions")

    # -------------------------------------------------------------------------
    # Internal forward helper
    # -------------------------------------------------------------------------
    async def _forward(body: dict, wrapper: Wrapper, *, kind: str):
        if body.get("stream", False) and kind in Wrapper._STREAMABLE_KINDS:
            try:
                handle = await wrapper.open_stream(body, kind=kind)
            except ApiError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status)
            except Exception as exc:  # pragma: no cover - defensive
                return _error_response(500, "Internal error", "server_error")
            return StreamingResponse(
                handle.lines(), media_type="text/event-stream", headers=ERROR_HEADERS
            )

        response = await wrapper.complete(body, kind=kind)
        return JSONResponse(content=response.body, status_code=response.status)

    return app
