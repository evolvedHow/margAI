"""Provider interface and OpenAI-compatible provider.

A provider owns three jobs:
  1. ``prepare_chat`` -- turn the internal (OpenAI-shaped) body into an
     upstream :class:`PreparedRequest` (URL, headers, native payload).
  2. ``parse_response`` / ``parse_chunk`` -- turn native responses back into
     the OpenAI shape the framework (and its hooks) operate on.
  3. ``list_models`` -- return upstream model ids for the catalog.
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

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
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
    # Providers that don't support the legacy completions surface inherit the
    # 404-raising fallbacks below, so `/v1/completions` degrades cleanly.

    def prepare_completions(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/completions",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_completion_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/completions", error_type="invalid_request_error")

    def parse_completion_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        raise ApiError(404, "Provider does not support /v1/completions", error_type="invalid_request_error")

    # -- embeddings -----------------------------------------------------------

    def prepare_embeddings(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/embeddings",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_embeddings_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/embeddings", error_type="invalid_request_error")

    # -- audio ----------------------------------------------------------------

    def prepare_audio_transcriptions(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/audio/transcriptions",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_audio_transcriptions_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/audio/transcriptions", error_type="invalid_request_error")

    def prepare_audio_translations(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/audio/translations",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_audio_translations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/audio/translations", error_type="invalid_request_error")

    def prepare_audio_speech(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/audio/speech",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_audio_speech_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/audio/speech", error_type="invalid_request_error")

    # -- images ---------------------------------------------------------------

    def prepare_images_generations(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/images/generations",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_images_generations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/images/generations", error_type="invalid_request_error")

    def prepare_images_edits(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/images/edits",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_images_edits_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/images/edits", error_type="invalid_request_error")

    def prepare_images_variations(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/images/variations",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_images_variations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/images/variations", error_type="invalid_request_error")

    # -- moderations ----------------------------------------------------------

    def prepare_moderations(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/moderations",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_moderations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/moderations", error_type="invalid_request_error")

    # -- files ----------------------------------------------------------------

    def prepare_files(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/files",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_files_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/files", error_type="invalid_request_error")

    def prepare_files_delete(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/files delete",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_files_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/files delete", error_type="invalid_request_error")

    def prepare_files_content(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/files content",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_files_content_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/files content", error_type="invalid_request_error")

    # -- fine-tuning ----------------------------------------------------------

    def prepare_fine_tuning_jobs(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/fine-tuning/jobs",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_fine_tuning_jobs_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/fine-tuning/jobs", error_type="invalid_request_error")

    def prepare_fine_tuning_jobs_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/fine-tuning/jobs list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_fine_tuning_jobs_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/fine-tuning/jobs list", error_type="invalid_request_error")

    def prepare_fine_tuning_jobs_cancel(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/fine-tuning/jobs cancel",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_fine_tuning_jobs_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/fine-tuning/jobs cancel", error_type="invalid_request_error")

    def prepare_fine_tuning_events(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/fine-tuning/events",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_fine_tuning_events_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/fine-tuning/events", error_type="invalid_request_error")

    # -- batches --------------------------------------------------------------

    def prepare_batches(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/batches",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_batches_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/batches", error_type="invalid_request_error")

    def prepare_batches_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/batches retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_batches_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/batches retrieve", error_type="invalid_request_error")

    def prepare_batches_cancel(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/batches cancel",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_batches_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/batches cancel", error_type="invalid_request_error")

    def prepare_batches_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/batches list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_batches_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/batches list", error_type="invalid_request_error")

    # -- vector stores --------------------------------------------------------

    def prepare_vector_stores(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/vector_stores",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_vector_stores_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/vector_stores", error_type="invalid_request_error")

    def prepare_vector_stores_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/vector_stores retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_vector_stores_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/vector_stores retrieve", error_type="invalid_request_error")

    def prepare_vector_stores_delete(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/vector_stores delete",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_vector_stores_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/vector_stores delete", error_type="invalid_request_error")

    def prepare_vector_stores_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/vector_stores list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_vector_stores_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/vector_stores list", error_type="invalid_request_error")

    # -- assistants -----------------------------------------------------------

    def prepare_assistants(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/assistants",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_assistants_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/assistants", error_type="invalid_request_error")

    def prepare_assistants_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/assistants retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_assistants_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/assistants retrieve", error_type="invalid_request_error")

    def prepare_assistants_delete(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/assistants delete",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_assistants_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/assistants delete", error_type="invalid_request_error")

    def prepare_assistants_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/assistants list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_assistants_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/assistants list", error_type="invalid_request_error")

    # -- threads --------------------------------------------------------------

    def prepare_threads(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads", error_type="invalid_request_error")

    def prepare_threads_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads retrieve", error_type="invalid_request_error")

    def prepare_threads_delete(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads delete",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads delete", error_type="invalid_request_error")

    def prepare_threads_messages(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads messages",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_messages_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads messages", error_type="invalid_request_error")

    def prepare_threads_messages_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads messages list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_messages_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads messages list", error_type="invalid_request_error")

    def prepare_threads_messages_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads messages retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_messages_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads messages retrieve", error_type="invalid_request_error")

    def prepare_threads_runs(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs", error_type="invalid_request_error")

    def prepare_threads_runs_retrieve(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs retrieve",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs retrieve", error_type="invalid_request_error")

    def prepare_threads_runs_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs list", error_type="invalid_request_error")

    def prepare_threads_runs_cancel(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs cancel",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs cancel", error_type="invalid_request_error")

    def prepare_threads_runs_steps(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs steps",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_steps_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs steps", error_type="invalid_request_error")

    def prepare_threads_runs_steps_list(self, ctx: Any) -> PreparedRequest:
        raise ApiError(
            404,
            f"Provider '{self.name}' does not support /v1/threads runs steps list",
            error_type="invalid_request_error",
            param="model",
        )

    def parse_threads_runs_steps_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        raise ApiError(404, "Provider does not support /v1/threads runs steps list", error_type="invalid_request_error")

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
            timeout=self.config.timeout,
        )

    def parse_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"choices": []}), status

    def parse_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        return _parse_sse_data_line(raw_line)

    def prepare_completions(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("completions"),
            headers=self.headers(),
            json=ctx.body,  # already OpenAI-shaped; `model` already rewritten
            timeout=self.config.timeout,
        )

    def parse_completion_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"choices": []}), status

    def parse_completion_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        return _parse_sse_data_line(raw_line)

    # -- embeddings -----------------------------------------------------------

    def prepare_embeddings(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("embeddings"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_embeddings_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- audio ----------------------------------------------------------------

    def prepare_audio_transcriptions(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("audio", "transcriptions"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_audio_transcriptions_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"text": ""}), status

    def prepare_audio_translations(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("audio", "translations"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_audio_translations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"text": ""}), status

    def prepare_audio_speech(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("audio", "speech"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_audio_speech_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    # -- images ---------------------------------------------------------------

    def prepare_images_generations(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("images", "generations"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_images_generations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    def prepare_images_edits(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("images", "edits"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_images_edits_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    def prepare_images_variations(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("images", "variations"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_images_variations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- moderations ----------------------------------------------------------

    def prepare_moderations(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("moderations"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_moderations_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"results": []}), status

    # -- files ----------------------------------------------------------------

    def prepare_files(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("files"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_files_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_files_delete(self, ctx: Any) -> PreparedRequest:
        file_id = ctx.body.get("file_id", "")
        return PreparedRequest(
            method="DELETE",
            url=self.endpoint("files", file_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_files_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_files_content(self, ctx: Any) -> PreparedRequest:
        file_id = ctx.body.get("file_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("files", file_id, "content"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_files_content_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    # -- fine-tuning ----------------------------------------------------------

    def prepare_fine_tuning_jobs(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("fine-tuning", "jobs"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_fine_tuning_jobs_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_fine_tuning_jobs_list(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="GET",
            url=self.endpoint("fine-tuning", "jobs"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_fine_tuning_jobs_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    def prepare_fine_tuning_jobs_cancel(self, ctx: Any) -> PreparedRequest:
        job_id = ctx.body.get("job_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("fine-tuning", "jobs", job_id, "cancel"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_fine_tuning_jobs_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_fine_tuning_events(self, ctx: Any) -> PreparedRequest:
        job_id = ctx.body.get("job_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("fine-tuning", "jobs", job_id, "events"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_fine_tuning_events_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- batches --------------------------------------------------------------

    def prepare_batches(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("batches"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_batches_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_batches_retrieve(self, ctx: Any) -> PreparedRequest:
        batch_id = ctx.body.get("batch_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("batches", batch_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_batches_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_batches_cancel(self, ctx: Any) -> PreparedRequest:
        batch_id = ctx.body.get("batch_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("batches", batch_id, "cancel"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_batches_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_batches_list(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="GET",
            url=self.endpoint("batches"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_batches_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- vector stores --------------------------------------------------------

    def prepare_vector_stores(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("vector_stores"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_vector_stores_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_vector_stores_retrieve(self, ctx: Any) -> PreparedRequest:
        store_id = ctx.body.get("vector_store_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("vector_stores", store_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_vector_stores_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_vector_stores_delete(self, ctx: Any) -> PreparedRequest:
        store_id = ctx.body.get("vector_store_id", "")
        return PreparedRequest(
            method="DELETE",
            url=self.endpoint("vector_stores", store_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_vector_stores_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_vector_stores_list(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="GET",
            url=self.endpoint("vector_stores"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_vector_stores_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- assistants -----------------------------------------------------------

    def prepare_assistants(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("assistants"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_assistants_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_assistants_retrieve(self, ctx: Any) -> PreparedRequest:
        assistant_id = ctx.body.get("assistant_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("assistants", assistant_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_assistants_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_assistants_delete(self, ctx: Any) -> PreparedRequest:
        assistant_id = ctx.body.get("assistant_id", "")
        return PreparedRequest(
            method="DELETE",
            url=self.endpoint("assistants", assistant_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_assistants_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_assistants_list(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="GET",
            url=self.endpoint("assistants"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_assistants_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    # -- threads --------------------------------------------------------------

    def prepare_threads(self, ctx: Any) -> PreparedRequest:
        return PreparedRequest(
            method="POST",
            url=self.endpoint("threads"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_threads_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_retrieve(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_delete(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="DELETE",
            url=self.endpoint("threads", thread_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_delete_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_messages(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("threads", thread_id, "messages"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_threads_messages_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_messages_list(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id, "messages"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_messages_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    def prepare_threads_messages_retrieve(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        message_id = ctx.body.get("message_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id, "messages", message_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_messages_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_runs(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("threads", thread_id, "runs"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_threads_runs_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_runs_retrieve(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        run_id = ctx.body.get("run_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id, "runs", run_id),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_runs_retrieve_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_runs_list(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id, "runs"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_runs_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

    def prepare_threads_runs_cancel(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        run_id = ctx.body.get("run_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("threads", thread_id, "runs", run_id, "cancel"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_runs_cancel_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_runs_steps(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        run_id = ctx.body.get("run_id", "")
        return PreparedRequest(
            method="POST",
            url=self.endpoint("threads", thread_id, "runs", run_id, "steps"),
            headers=self.headers(),
            json=ctx.body,
            timeout=self.config.timeout,
        )

    def parse_threads_runs_steps_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {}), status

    def prepare_threads_runs_steps_list(self, ctx: Any) -> PreparedRequest:
        thread_id = ctx.body.get("thread_id", "")
        run_id = ctx.body.get("run_id", "")
        return PreparedRequest(
            method="GET",
            url=self.endpoint("threads", thread_id, "runs", run_id, "steps"),
            headers=self.headers(),
            timeout=self.config.timeout,
        )

    def parse_threads_runs_steps_list_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise _error_from_response(resp, status)
        return (resp.body if isinstance(resp.body, dict) else {"data": []}), status

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
        return json.loads(data)
    except ValueError:
        return None


def _error_from_response(resp: UpstreamResponse, status: int):
    return ApiError.from_openai_body(resp.body, implicit_status=status)