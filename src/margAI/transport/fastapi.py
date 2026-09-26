"""FastAPI adapter: expose a :class:`Wrapper` behind an OpenAI-compatible API.

This is the transport LibreChat (and any OpenAI SDK) points at. The wrapper
can be built from config, or injected directly (tests, custom wiring).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Union

from fastapi import FastAPI, Request, File, UploadFile, Form, Query, status, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict

from ..config import Config, load_config
from ..core.errors import ApiError
from ..wrapper import Wrapper

__all__ = ["build_app"]

ERROR_HEADERS = {"X-Accel-Buffering": "no"}


# =============================================================================
# Pydantic Models for OpenAI API - Chat Completions
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


class ChatCompletionChunkRequest(ChatCompletionRequest):
    stream: Literal[True] = True


# =============================================================================
# Completions (Legacy)
# =============================================================================

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


# =============================================================================
# Embeddings
# =============================================================================

class EmbeddingRequest(BaseModel):
    input: Union[str, list[str], list[int], list[list[int]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


# =============================================================================
# Audio
# =============================================================================

class AudioTranscriptionRequest(BaseModel):
    file: Any  # Will be handled by FastAPI's File()
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = "json"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None


class AudioTranslationRequest(BaseModel):
    file: Any
    model: str
    prompt: Optional[str] = None
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = "json"
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "mp3"
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)


# =============================================================================
# Images
# =============================================================================

class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[Literal["standard", "hd"]] = "standard"
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] = "1024x1024"
    style: Optional[Literal["vivid", "natural"]] = "vivid"
    user: Optional[str] = None


class ImageEditRequest(BaseModel):
    image: Any  # File
    model: str
    prompt: str
    mask: Optional[Any] = None  # File
    n: Optional[int] = Field(default=1, ge=1, le=10)
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = "1024x1024"
    user: Optional[str] = None


class ImageVariationRequest(BaseModel):
    image: Any  # File
    model: str
    n: Optional[int] = Field(default=1, ge=1, le=10)
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = "1024x1024"
    user: Optional[str] = None


# =============================================================================
# Moderations
# =============================================================================

class ModerationRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = None


# =============================================================================
# Files
# =============================================================================

class FileUploadRequest(BaseModel):
    file: Any  # File
    purpose: Literal["fine-tune", "assistants", "batch", "vision"]


# =============================================================================
# Fine-tuning
# =============================================================================

class FineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    hyperparameters: Optional[dict[str, Any]] = None
    integration: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    suffix: Optional[str] = None
    validation_file: Optional[str] = None


class FineTuningJobListRequest(BaseModel):
    after: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)


class FineTuningJobCancelRequest(BaseModel):
    job_id: str


class FineTuningEventRequest(BaseModel):
    job_id: str
    after: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)


# =============================================================================
# Batches
# =============================================================================

class BatchRequest(BaseModel):
    input_file_id: str
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"]
    completion_window: Literal["24h"]
    metadata: Optional[dict[str, str]] = None


class BatchRetrieveRequest(BaseModel):
    batch_id: str


class BatchCancelRequest(BaseModel):
    batch_id: str


class BatchListRequest(BaseModel):
    after: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)


# =============================================================================
# Vector Stores
# =============================================================================

class VectorStoreRequest(BaseModel):
    name: Optional[str] = None
    file_ids: Optional[list[str]] = None
    chunking_strategy: Optional[dict[str, Any]] = None
    expires_after: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, str]] = None


class VectorStoreRetrieveRequest(BaseModel):
    vector_store_id: str


class VectorStoreDeleteRequest(BaseModel):
    vector_store_id: str


class VectorStoreListRequest(BaseModel):
    after: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    order: Optional[Literal["asc", "desc"]] = "desc"


# =============================================================================
# Assistants
# =============================================================================

class AssistantRequest(BaseModel):
    model: str
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_resources: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, str]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    response_format: Optional[Union[Literal["auto"], dict[str, Any]]] = "auto"


class AssistantRetrieveRequest(BaseModel):
    assistant_id: str


class AssistantDeleteRequest(BaseModel):
    assistant_id: str


class AssistantListRequest(BaseModel):
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    order: Optional[Literal["asc", "desc"]] = "desc"


# =============================================================================
# Threads
# =============================================================================

class ThreadRequest(BaseModel):
    messages: Optional[list[dict[str, Any]]] = None
    tool_resources: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, str]] = None


class ThreadRetrieveRequest(BaseModel):
    thread_id: str


class ThreadDeleteRequest(BaseModel):
    thread_id: str


class ThreadMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[dict[str, Any]]]
    attachments: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, str]] = None


class ThreadMessageListRequest(BaseModel):
    thread_id: str
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    order: Optional[Literal["asc", "desc"]] = "desc"
    run_id: Optional[str] = None


class ThreadMessageRetrieveRequest(BaseModel):
    thread_id: str
    message_id: str


class ThreadRunRequest(BaseModel):
    thread_id: str
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    additional_messages: Optional[list[dict[str, Any]]] = None
    additional_tools: Optional[list[dict[str, Any]]] = None
    tool_resources: Optional[dict[str, Any]] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[dict[str, Any]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None


class ThreadRunRetrieveRequest(BaseModel):
    thread_id: str
    run_id: str


class ThreadRunListRequest(BaseModel):
    thread_id: str
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    order: Optional[Literal["asc", "desc"]] = "desc"


class ThreadRunCancelRequest(BaseModel):
    thread_id: str
    run_id: str


class ThreadRunStepsListRequest(BaseModel):
    thread_id: str
    run_id: str
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = Field(default=20, ge=1, le=100)
    order: Optional[Literal["asc", "desc"]] = "desc"


class ThreadRunStepRequest(BaseModel):
    thread_id: str
    run_id: str
    step_id: str


# =============================================================================
# Models
# =============================================================================

class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelResponse]


# =============================================================================
# Error Response
# =============================================================================

class ErrorResponse(BaseModel):
    error: dict[str, Any]


# =============================================================================
# Request Body Helpers
# =============================================================================

async def _request_body(request: Request) -> tuple[dict | None, JSONResponse | None]:
    try:
        body = await request.json()
    except Exception:
        return None, _error_response(400, "Request body must be valid JSON")
    if not isinstance(body, dict):
        return None, _error_response(400, "Request body must be a JSON object")
    return body, None


def _error_response(status: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


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
        return _error_response(400, "Request body must be valid JSON", "invalid_request_error")

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
        models = await wrapper.models()
        for model in models:
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
    # Audio
    # -------------------------------------------------------------------------
    @app.post("/v1/audio/transcriptions", tags=["Audio"])
    async def audio_transcriptions(body: AudioTranscriptionRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="audio_transcriptions")

    @app.post("/v1/audio/translations", tags=["Audio"])
    async def audio_translations(body: AudioTranslationRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="audio_translations")

    @app.post("/v1/audio/speech", tags=["Audio"])
    async def audio_speech(body: AudioSpeechRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="audio_speech")

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------
    @app.post("/v1/images/generations", tags=["Images"])
    async def images_generations(body: ImageGenerationRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="images_generations")

    @app.post("/v1/images/edits", tags=["Images"])
    async def images_edits(body: ImageEditRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="images_edits")

    @app.post("/v1/images/variations", tags=["Images"])
    async def images_variations(body: ImageVariationRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="images_variations")

    # -------------------------------------------------------------------------
    # Moderations
    # -------------------------------------------------------------------------
    @app.post("/v1/moderations", tags=["Moderations"])
    async def moderations(body: ModerationRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="moderations")

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------
    @app.post("/v1/files", tags=["Files"])
    async def create_file(body: FileUploadRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="files")

    @app.get("/v1/files", tags=["Files"])
    async def list_files(request: Request):
        body = {"limit": request.query_params.get("limit", 20), "after": request.query_params.get("after")}
        return await _forward(body, wrapper, kind="files")

    @app.get("/v1/files/{file_id}", tags=["Files"])
    async def retrieve_file(file_id: str):
        body = {"file_id": file_id}
        return await _forward(body, wrapper, kind="files_content")

    @app.delete("/v1/files/{file_id}", tags=["Files"])
    async def delete_file(file_id: str):
        body = {"file_id": file_id}
        return await _forward(body, wrapper, kind="files_delete")

    @app.get("/v1/files/{file_id}/content", tags=["Files"])
    async def retrieve_file_content(file_id: str):
        body = {"file_id": file_id}
        return await _forward(body, wrapper, kind="files_content")

    # -------------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------------
    @app.post("/v1/fine-tuning/jobs", tags=["Fine-tuning"])
    async def create_fine_tuning_job(body: FineTuningJobRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="fine_tuning_jobs")

    @app.get("/v1/fine-tuning/jobs", tags=["Fine-tuning"])
    async def list_fine_tuning_jobs(request: Request):
        body = {
            "after": request.query_params.get("after"),
            "limit": request.query_params.get("limit", 20),
        }
        return await _forward(body, wrapper, kind="fine_tuning_jobs_list")

    @app.get("/v1/fine-tuning/jobs/{job_id}", tags=["Fine-tuning"])
    async def retrieve_fine_tuning_job(job_id: str):
        body = {"job_id": job_id}
        return await _forward(body, wrapper, kind="fine_tuning_jobs")

    @app.post("/v1/fine-tuning/jobs/{job_id}/cancel", tags=["Fine-tuning"])
    async def cancel_fine_tuning_job(job_id: str):
        body = {"job_id": job_id}
        return await _forward(body, wrapper, kind="fine_tuning_jobs_cancel")

    @app.get("/v1/fine-tuning/jobs/{job_id}/events", tags=["Fine-tuning"])
    async def list_fine_tuning_events(job_id: str, request: Request):
        body = {
            "job_id": job_id,
            "after": request.query_params.get("after"),
            "limit": request.query_params.get("limit", 20),
        }
        return await _forward(body, wrapper, kind="fine_tuning_events")

    # -------------------------------------------------------------------------
    # Batches
    # -------------------------------------------------------------------------
    @app.post("/v1/batches", tags=["Batches"])
    async def create_batch(body: BatchRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="batches")

    @app.get("/v1/batches", tags=["Batches"])
    async def list_batches(request: Request):
        body = {
            "after": request.query_params.get("after"),
            "limit": request.query_params.get("limit", 20),
        }
        return await _forward(body, wrapper, kind="batches_list")

    @app.get("/v1/batches/{batch_id}", tags=["Batches"])
    async def retrieve_batch(batch_id: str):
        body = {"batch_id": batch_id}
        return await _forward(body, wrapper, kind="batches_retrieve")

    @app.post("/v1/batches/{batch_id}/cancel", tags=["Batches"])
    async def cancel_batch(batch_id: str):
        body = {"batch_id": batch_id}
        return await _forward(body, wrapper, kind="batches_cancel")

    # -------------------------------------------------------------------------
    # Vector Stores
    # -------------------------------------------------------------------------
    @app.post("/v1/vector_stores", tags=["Vector Stores"])
    async def create_vector_store(body: VectorStoreRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="vector_stores")

    @app.get("/v1/vector_stores", tags=["Vector Stores"])
    async def list_vector_stores(request: Request):
        body = {
            "after": request.query_params.get("after"),
            "limit": request.query_params.get("limit", 20),
            "order": request.query_params.get("order", "desc"),
        }
        return await _forward(body, wrapper, kind="vector_stores_list")

    @app.get("/v1/vector_stores/{vector_store_id}", tags=["Vector Stores"])
    async def retrieve_vector_store(vector_store_id: str):
        body = {"vector_store_id": vector_store_id}
        return await _forward(body, wrapper, kind="vector_stores_retrieve")

    @app.delete("/v1/vector_stores/{vector_store_id}", tags=["Vector Stores"])
    async def delete_vector_store(vector_store_id: str):
        body = {"vector_store_id": vector_store_id}
        return await _forward(body, wrapper, kind="vector_stores_delete")

    # -------------------------------------------------------------------------
    # Assistants
    # -------------------------------------------------------------------------
    @app.post("/v1/assistants", tags=["Assistants"])
    async def create_assistant(body: AssistantRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="assistants")

    @app.get("/v1/assistants", tags=["Assistants"])
    async def list_assistants(request: Request):
        body = {
            "after": request.query_params.get("after"),
            "before": request.query_params.get("before"),
            "limit": request.query_params.get("limit", 20),
            "order": request.query_params.get("order", "desc"),
        }
        return await _forward(body, wrapper, kind="assistants_list")

    @app.get("/v1/assistants/{assistant_id}", tags=["Assistants"])
    async def retrieve_assistant(assistant_id: str):
        body = {"assistant_id": assistant_id}
        return await _forward(body, wrapper, kind="assistants_retrieve")

    @app.delete("/v1/assistants/{assistant_id}", tags=["Assistants"])
    async def delete_assistant(assistant_id: str):
        body = {"assistant_id": assistant_id}
        return await _forward(body, wrapper, kind="assistants_delete")

    @app.post("/v1/assistants/{assistant_id}", tags=["Assistants"])
    async def modify_assistant(assistant_id: str, body: AssistantRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["assistant_id"] = assistant_id
        return await _forward(data, wrapper, kind="assistants")

    # -------------------------------------------------------------------------
    # Threads
    # -------------------------------------------------------------------------
    @app.post("/v1/threads", tags=["Threads"])
    async def create_thread(body: ThreadRequest = Body(...)):
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="threads")

    @app.get("/v1/threads/{thread_id}", tags=["Threads"])
    async def retrieve_thread(thread_id: str):
        body = {"thread_id": thread_id}
        return await _forward(body, wrapper, kind="threads_retrieve")

    @app.delete("/v1/threads/{thread_id}", tags=["Threads"])
    async def delete_thread(thread_id: str):
        body = {"thread_id": thread_id}
        return await _forward(body, wrapper, kind="threads_delete")

    @app.post("/v1/threads/{thread_id}", tags=["Threads"])
    async def modify_thread(thread_id: str, body: ThreadRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["thread_id"] = thread_id
        return await _forward(data, wrapper, kind="threads")

    # Thread Messages
    @app.post("/v1/threads/{thread_id}/messages", tags=["Threads"])
    async def create_thread_message(thread_id: str, body: ThreadMessageRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["thread_id"] = thread_id
        return await _forward(data, wrapper, kind="threads_messages")

    @app.get("/v1/threads/{thread_id}/messages", tags=["Threads"])
    async def list_thread_messages(thread_id: str, request: Request):
        body = {
            "thread_id": thread_id,
            "after": request.query_params.get("after"),
            "before": request.query_params.get("before"),
            "limit": request.query_params.get("limit", 20),
            "order": request.query_params.get("order", "desc"),
            "run_id": request.query_params.get("run_id"),
        }
        return await _forward(body, wrapper, kind="threads_messages_list")

    @app.get("/v1/threads/{thread_id}/messages/{message_id}", tags=["Threads"])
    async def retrieve_thread_message(thread_id: str, message_id: str):
        body = {"thread_id": thread_id, "message_id": message_id}
        return await _forward(body, wrapper, kind="threads_messages_retrieve")

    @app.post("/v1/threads/{thread_id}/messages/{message_id}", tags=["Threads"])
    async def modify_thread_message(thread_id: str, message_id: str, body: ThreadMessageRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["thread_id"] = thread_id
        data["message_id"] = message_id
        return await _forward(data, wrapper, kind="threads_messages")

    # Thread Runs
    @app.post("/v1/threads/{thread_id}/runs", tags=["Threads"])
    async def create_thread_run(thread_id: str, body: ThreadRunRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["thread_id"] = thread_id
        return await _forward(data, wrapper, kind="threads_runs")

    @app.get("/v1/threads/{thread_id}/runs", tags=["Threads"])
    async def list_thread_runs(thread_id: str, request: Request):
        body = {
            "thread_id": thread_id,
            "after": request.query_params.get("after"),
            "before": request.query_params.get("before"),
            "limit": request.query_params.get("limit", 20),
            "order": request.query_params.get("order", "desc"),
        }
        return await _forward(body, wrapper, kind="threads_runs_list")

    @app.get("/v1/threads/{thread_id}/runs/{run_id}", tags=["Threads"])
    async def retrieve_thread_run(thread_id: str, run_id: str):
        body = {"thread_id": thread_id, "run_id": run_id}
        return await _forward(body, wrapper, kind="threads_runs_retrieve")

    @app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel", tags=["Threads"])
    async def cancel_thread_run(thread_id: str, run_id: str):
        body = {"thread_id": thread_id, "run_id": run_id}
        return await _forward(body, wrapper, kind="threads_runs_cancel")

    @app.post("/v1/threads/{thread_id}/runs/{run_id}", tags=["Threads"])
    async def modify_thread_run(thread_id: str, run_id: str, body: ThreadRunRequest = Body(...)):
        data = body.model_dump(exclude_none=True)
        data["thread_id"] = thread_id
        data["run_id"] = run_id
        return await _forward(data, wrapper, kind="threads_runs")

    # Thread Run Steps
    @app.get("/v1/threads/{thread_id}/runs/{run_id}/steps", tags=["Threads"])
    async def list_thread_run_steps(thread_id: str, run_id: str, request: Request):
        body = {
            "thread_id": thread_id,
            "run_id": run_id,
            "after": request.query_params.get("after"),
            "before": request.query_params.get("before"),
            "limit": request.query_params.get("limit", 20),
            "order": request.query_params.get("order", "desc"),
        }
        return await _forward(body, wrapper, kind="threads_runs_steps_list")

    @app.get("/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}", tags=["Threads"])
    async def retrieve_thread_run_step(thread_id: str, run_id: str, step_id: str):
        body = {"thread_id": thread_id, "run_id": run_id, "step_id": step_id}
        return await _forward(body, wrapper, kind="threads_runs_steps")

    # -------------------------------------------------------------------------
    # Internal forward helper
    # -------------------------------------------------------------------------
    async def _forward(body: dict, wrapper: Wrapper, *, kind: str):
        if body.get("stream", False) and kind in ("chat", "completions"):
            try:
                handle = await wrapper.open_stream(body, kind=kind)
            except ApiError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status)
            except Exception as exc:  # pragma: no cover - defensive
                return _error_response(500, str(exc), "server_error")
            return StreamingResponse(
                handle.lines(), media_type="text/event-stream", headers=ERROR_HEADERS
            )

        response = await wrapper.complete(body, kind=kind)
        return JSONResponse(content=response.body, status_code=response.status)

    return app