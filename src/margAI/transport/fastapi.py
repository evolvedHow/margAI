"""FastAPI adapter: expose a :class:`Wrapper` behind an OpenAI-compatible API.

This is the transport LibreChat (and any OpenAI SDK) points at. The wrapper
can be built from config, or injected directly (tests, custom wiring).

The routed surface is chat, legacy completions, embeddings, image
generation, and audio transcription -- see ``docs/ENDPOINTS.md``.

Every request field, response shape, and error status is described here, so the
generated OpenAPI document *is* the reference: ``/docs`` (Swagger UI),
``/redoc`` (ReDoc) and ``/openapi.json`` are all generated from the models and
route metadata below. A parameter whose meaning is not written down here is a
parameter the docs get wrong.

Response models are attached with ``responses=`` rather than
``response_model=`` on purpose. Every route returns a ``JSONResponse`` -- or a
``StreamingResponse`` when ``stream=true`` -- so hooks can shape the body and
upstream status freely. ``response_model=`` would interpose a validation and
serialization step, silently dropping any upstream field the model does not
name. ``responses=`` documents the same schema without touching the runtime.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config import Config, load_config
from ..core.errors import ApiError
from ..wrapper import Wrapper

__all__ = ["build_app"]

ERROR_HEADERS = {"X-Accel-Buffering": "no"}

#: Written once, reused by every field that names a model, so the accepted
#: spellings cannot drift between endpoints.
_MODEL_FIELD_HELP = (
    "Model id to serve this call. Accepted forms: a namespaced id as "
    "`GET /v1/models` lists them, e.g. `<namespace>/<provider>/<model>`; a bare "
    "`<model>` resolved against every provider's configured `models` list; "
    "`<namespace>/dynamic`, the reserved id whose target is chosen per call by "
    "the routing chain after all `before` hooks have run; or "
    "`<namespace>/<name>` for a virtual model declared under `[models.*]`. An "
    "unresolvable id is a 404 that names the ids that do resolve. "
    "`GET /v1/models` only *lists* ids, per `gateway.expose`; that setting "
    "never gates what this field accepts."
)

#: The bangtag directive, described once. `<namespace>` is the gateway's
#: configured `gateway.prefix`; the literal prefix is substituted into the
#: app-level description, which is assembled where the prefix is known.
_BANGTAG_HELP = (
    "A leading `!<namespace>: <tag>[, <tag>=<value>]` directive is parsed out of "
    "the last user message and stripped from the text before the prompt reaches "
    "the provider, so a directive never appears in the model's context. "
    "`<namespace>` is this gateway's `gateway.prefix`. "
    "`GET /v1/margAI/tags` reports the tags actually registered; an "
    "unrecognized tag is dropped with a warning, and the directive is stripped "
    "either way."
)


# =============================================================================
# Pydantic request models
# =============================================================================

class ChatCompletionMessageParam(BaseModel):
    """One message in a chat conversation."""

    role: Literal["system", "user", "assistant", "developer", "function", "tool"] = Field(
        description=(
            "Who is speaking. `system` and `developer` set instructions; `user` "
            "and `assistant` carry the conversation; `function` and `tool` carry "
            "a tool result back to the model."
        )
    )
    content: str | list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Message text, or an array of content parts for multimodal input. "
            f"On the last `user` message: {_BANGTAG_HELP}"
        ),
    )
    name: str | None = Field(
        default=None,
        description="Author name, on a `function` or `tool` message.",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool calls the assistant is requesting. Present on an `assistant` "
            "message in a tool-calling turn; the client answers with a `tool` "
            "message carrying the matching `tool_call_id`."
        ),
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Which `tool_calls` entry this `tool` message answers.",
    )
    function_call: dict[str, Any] | None = Field(
        default=None,
        description="Deprecated single-function call shape, superseded by `tools`.",
    )


class ChatCompletionToolParam(BaseModel):
    """A tool the model is allowed to call."""

    type: Literal["function"] = Field(
        default="function",
        description="Tool kind. Only `function` exists on the OpenAI-compatible surface.",
    )
    function: dict[str, Any] = Field(
        description=(
            "Function definition, OpenAI-shaped: a `name` plus a JSON Schema "
            "`parameters` object describing the accepted arguments."
        )
    )


class ChatCompletionRequest(BaseModel):
    """Body for `POST /v1/chat/completions`."""

    model: str = Field(description=_MODEL_FIELD_HELP)
    messages: list[ChatCompletionMessageParam] = Field(
        description=(
            "The conversation so far, oldest first, with at least one message. "
            "`before` hooks and any bangtag handlers this call activates may "
            "add, rewrite or reorder these before the provider sees them."
        )
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalizes tokens by how often they have already appeared. Negative "
            "encourages reuse, positive discourages it. Range -2 to 2."
        ),
    )
    logit_bias: dict[str, int] | None = Field(
        default=None,
        description=(
            "Per-token bias applied before sampling, keyed by token id as a "
            "string. Values run roughly -100 to 100 and shift the odds of a "
            "token being produced; most providers honour only a subset of tokens."
        ),
    )
    logprobs: bool | None = Field(
        default=None,
        description="Return per-token log probabilities alongside each choice.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description=(
            "How many of the most likely tokens to report log probabilities for, "
            "at each position. Only meaningful when `logprobs` is true."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Upper bound on tokens generated. Prefer `max_completion_tokens`: "
            "`max_tokens` is deprecated on reasoning models, which count their "
            "reasoning tokens against it."
        ),
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Upper bound on generated tokens, reasoning included. Takes "
            "precedence over `max_tokens` when both are sent."
        ),
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=128,
        description="How many independent completions to generate for this prompt.",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalizes tokens that have appeared at all. Range -2 to 2; use "
            "`frequency_penalty` to scale with frequency instead."
        ),
    )
    response_format: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Output shape, OpenAI-shaped: `{\"type\": \"text\"}` or "
            "`{\"type\": \"json_object\"}`. `json_schema` variants pass through "
            "unchanged. Support varies by model."
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Seed for deterministic sampling. Best-effort: a provider may still "
            "vary output for the same seed, and one seed does not imply "
            "reproducibility across model versions."
        ),
    )
    stop: str | list[str] | None = Field(
        default=None,
        description=(
            "A stop sequence, or up to four of them. Generation halts when one is "
            "produced. Never echoed back in the response."
        ),
    )
    stream: bool | None = Field(
        default=False,
        description=(
            "Stream the response as server-sent events. Each `data:` frame is a "
            "chunk with the same envelope as the non-streaming body but a `delta` "
            "in place of `message`, terminated by `data: [DONE]`. The response "
            "`Content-Type` becomes `text/event-stream`."
        ),
    )
    stream_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Streaming options. `{\"include_usage\": true}` appends one final "
            "chunk carrying a `usage` object and no choices, so a stream reports "
            "the same token counts a non-streaming call would."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature, 0 to 2. Near 0 is near-deterministic; higher "
            "values sample more broadly. Unsupported by some reasoning models."
        ),
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Nucleus sampling: consider only the smallest set of tokens whose "
            "probability sums to at least this value, 0 to 1. An alternative to "
            "`temperature`, not a complement to it."
        ),
    )
    tools: list[ChatCompletionToolParam] | None = Field(
        default=None,
        description="Tools the model may call during this turn.",
    )
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Which tool to call: `none`, `auto` (the model decides), `required` "
            "(it must call one), or an object naming a specific function."
        ),
    )
    user: str | None = Field(
        default=None,
        description=(
            "Opaque end-user identifier. Providers use it for abuse monitoring "
            "and may reject `null`. margAI records it on the telemetry row."
        ),
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Allow the model to request more than one tool call per turn.",
    )


class CompletionRequest(BaseModel):
    """Body for `POST /v1/completions` -- the legacy text-completion surface.

    Prefer `POST /v1/chat/completions` for new work: this endpoint has no
    concept of a system message, and not every provider serves it.
    """

    model: str = Field(description=_MODEL_FIELD_HELP)
    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        description=(
            "Text to complete. One string, a list of strings for batched prompts, "
            "a list of token ids, or a list of token-id lists. Each prompt in a "
            "batch yields its own choice."
        )
    )
    best_of: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Generate this many completions server-side and keep the best `n`. "
            "Only meaningful when `n` is above 1, and unsupported by some providers."
        ),
    )
    echo: bool | None = Field(
        default=False,
        description="Include the original prompt in the returned text.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalizes tokens by how often they have already appeared. Range -2 to 2."
        ),
    )
    logit_bias: dict[str, int] | None = Field(
        default=None,
        description="Per-token bias applied before sampling, keyed by token id as a string.",
    )
    logprobs: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description=(
            "Number of most likely tokens to return log probabilities for, at each "
            "position. 0 means the sampled token only."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound on tokens generated.",
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=128,
        description="How many completions to generate for each prompt.",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes tokens that have appeared at all. Range -2 to 2.",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for deterministic sampling. Best-effort, as on chat.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="A stop sequence, or up to four of them.",
    )
    stream: bool | None = Field(
        default=False,
        description=(
            "Stream as server-sent events. Each `data:` frame carries a chunk whose "
            "choice holds incremental `text` rather than a whole completion, "
            "terminated by `data: [DONE]`."
        ),
    )
    stream_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Streaming options. `{\"include_usage\": true}` appends a final chunk "
            "carrying a `usage` object and no choices."
        ),
    )
    suffix: str | None = Field(
        default=None,
        description=(
            "Text appended after the completion, for insertion-style models. Not "
            "supported by chat models."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature, 0 to 2.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold, 0 to 1.",
    )
    user: str | None = Field(
        default=None,
        description="Opaque end-user identifier, recorded on the telemetry row.",
    )


class EmbeddingRequest(BaseModel):
    """Body for `POST /v1/embeddings`."""

    model: str = Field(description=_MODEL_FIELD_HELP)
    input: str | list[str] | list[int] | list[list[int]] = Field(
        description=(
            "Text to embed, a list of texts, token ids, or token-id lists. Every "
            "input yields one vector, in order. There is no `before`/`after` tag "
            "dispatch difference here: `before` hooks still run, and `after` hooks "
            "still shape the vectors, but this kind never streams."
        )
    )
    encoding_format: Literal["float", "base64"] | None = Field(
        default="float",
        description=(
            "`float` returns an array of numbers. `base64` returns a base64 "
            "encoding of the little-endian float32 array, which is smaller on the "
            "wire; decode it client-side. Ignored by providers that return only one form."
        ),
    )
    dimensions: int | None = Field(
        default=None,
        description=(
            "Truncate the output to this many dimensions. Models that support "
            "Matryoshka-style shortening honour it; others ignore it or reject it, "
            "in which case a shortened vector would silently not match. Verify "
            "against the model before relying on it."
        ),
    )
    user: str | None = Field(
        default=None,
        description="Opaque end-user identifier, recorded on the telemetry row.",
    )


class ImageGenerationRequest(BaseModel):
    """Body for `POST /v1/images/generations`."""

    model: str = Field(description=_MODEL_FIELD_HELP)
    prompt: str = Field(
        description="What to draw. This is the entire creative brief; the model sees nothing else."
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=10,
        description="How many images to generate.",
    )
    quality: Literal["standard", "hd"] | None = Field(
        default="standard",
        description=(
            "`standard` is faster and cheaper; `hd` renders larger and more "
            "detailed. Provider-specific, and absent on some models."
        ),
    )
    response_format: Literal["url", "b64_json"] | None = Field(
        default="url",
        description=(
            "`url` returns a URL with a finite lifetime -- download it rather than "
            "storing it. `b64_json` returns the bytes inline, which avoids that "
            "expiry at the cost of a much larger body."
        ),
    )
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] | None = Field(
        default="1024x1024",
        description=(
            "Output dimensions as `WIDTHxHEIGHT`. Non-square options are only "
            "available on image models that support them."
        ),
    )
    style: Literal["vivid", "natural"] | None = Field(
        default="vivid",
        description=(
            "`vivid` pushes toward hyperreal and dramatic; `natural` toward "
            "literal interpretation of the prompt."
        ),
    )
    user: str | None = Field(
        default=None,
        description="Opaque end-user identifier, recorded on the telemetry row.",
    )


class AudioTranscriptionRequest(BaseModel):
    """The non-file fields of a `multipart/form-data` transcription.

    Unlike the other request models this one is not sent as a JSON body. Every
    field here is a form part alongside the required `file` upload, and the
    route documents the upload itself separately.
    """

    model: str = Field(description=_MODEL_FIELD_HELP)
    language: str | None = Field(
        default=None,
        description=(
            "ISO-639-1 language code of the audio, e.g. `en` or `fr`. Supplying it "
            "improves accuracy and avoids spending detection budget; omitting it "
            "makes the model detect the language itself."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "Optional context to steer the transcription -- domain vocabulary, the "
            "names of speakers, or an earlier fragment. Not a general instruction."
        ),
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = Field(
        default="json",
        description=(
            "`json` returns `{\"text\": ...}`. `verbose_json` adds segments, "
            "language and per-word timings. `text` returns a bare string; `srt` and "
            "`vtt` return subtitle formats. Note that only the structured formats "
            "carry `usage`, so a `text` or `srt` response cannot be costed from the "
            "body alone."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature, 0 to 1. 0 is the most literal transcription.",
    )
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        default=None,
        description=(
            "Which timestamp levels to include in a `verbose_json` response. Omitted "
            "means the provider's default. Requires `response_format=verbose_json`."
        ),
    )


# =============================================================================
# Response models
# =============================================================================
#
# Documentation only -- see the module docstring on why these are attached via
# `responses=` and never `response_model=`. They are all `extra="allow"` because
# the bodies genuinely are passthrough: whatever the upstream sent, minus
# anything a hook deleted. Naming only the fields margAI relies on is an honest
# floor, not an exhaustive schema, and `additionalProperties: true` is what
# tells a reader so.

class _Passthrough(BaseModel):
    """Base for the response models: unknown upstream fields are allowed."""

    model_config = ConfigDict(extra="allow")


class ErrorDetail(_Passthrough):
    """The `error` object margAI returns on every non-2xx response."""

    message: str = Field(
        description=(
            "Human-readable cause. Safe to show a user: an unhandled exception is "
            "never surfaced here, only a generic message with the detail in the log."
        )
    )
    type: str = Field(
        default="invalid_request_error",
        description=(
            "Error class. `invalid_request_error` for a malformed or rejected "
            "request, `not_found_error` for an unresolvable model, and the "
            "provider's own type when the failure came from upstream."
        ),
    )
    param: str | None = Field(
        default=None, description="The request field at fault, when one specific field is."
    )
    code: str | None = Field(
        default=None, description="Provider-specific error code, when one was supplied."
    )


class ErrorResponse(_Passthrough):
    """Every error body on this surface has exactly this shape."""

    error: ErrorDetail


class Usage(_Passthrough):
    """Token accounting for a call."""

    prompt_tokens: int = Field(description="Tokens in the request.", ge=0)
    completion_tokens: int = Field(
        description="Tokens generated. 0 for kinds that generate nothing, e.g. embeddings.",
        ge=0,
    )
    total_tokens: int = Field(
        description="`prompt_tokens` plus `completion_tokens`.", ge=0
    )


class ChatCompletionMessage(_Passthrough):
    """The assistant turn returned for a non-streaming chat call."""

    role: str = Field(default="assistant", description="Always `assistant` in a response.")
    content: str | None = Field(
        default=None,
        description=(
            "Generated text. `null` when the turn is a tool call rather than an "
            "answer -- check `tool_calls` before assuming there is nothing to show."
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls requested by the model, when `tools` was sent."
    )
    function_call: dict[str, Any] | None = Field(
        default=None, description="Deprecated single-function call shape."
    )


class ChatCompletionChoice(_Passthrough):
    """One candidate completion."""

    index: int = Field(description="Position of this choice in the `n` returned.", ge=0)
    message: ChatCompletionMessage = Field(description="The generated turn.")
    finish_reason: str | None = Field(
        default=None,
        description=(
            "Why generation stopped: `stop`, `length` (hit `max_tokens`), "
            "`tool_calls`, `content_filter`, or a provider-specific value. `null` "
            "while a stream is still open."
        ),
    )
    logprobs: dict[str, Any] | None = Field(
        default=None, description="Per-token log probabilities, when `logprobs` was requested."
    )


class ChatCompletionResponse(_Passthrough):
    """Body for a non-streaming `POST /v1/chat/completions`."""

    id: str = Field(description="Unique id for this completion, e.g. `chatcmpl-...`.")
    object: str = Field(default="chat.completion", description="Always `chat.completion`.")
    created: int = Field(description="Unix timestamp the completion was created at.", ge=0)
    model: str = Field(
        description=(
            "The model that actually served the call. This is the *upstream* id, "
            "not the `<namespace>/<provider>/<model>` you asked for, so it is what "
            "telemetry is attributed to."
        )
    )
    choices: list[ChatCompletionChoice] = Field(
        description="The generated candidates, in the order requested."
    )
    usage: Usage | None = Field(
        default=None, description="Token counts. Absent when the provider omits them or the call failed."
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="Backend configuration fingerprint, when the provider supplies one.",
    )


class ChatCompletionStreamChunk(_Passthrough):
    """One `data:` frame of a streaming chat call.

    Identical to :class:`ChatCompletionResponse` except that each choice holds a
    `delta` -- the incremental change -- instead of a whole `message`. The final
    frame has an empty delta and a `finish_reason`; with
    `stream_options.include_usage` a last frame after that carries only `usage`.
    """

    id: str = Field(description="Same `id` across every frame of one stream.")
    object: str = Field(
        default="chat.completion.chunk", description="Always `chat.completion.chunk`."
    )
    created: int = Field(description="Unix timestamp the stream was created at.", ge=0)
    model: str = Field(description="The model serving the stream.")
    choices: list[dict[str, Any]] = Field(
        description=(
            "Frames with `delta: {\"content\": ...}`, ending in an empty delta with "
            "a `finish_reason`. The usage-only frame that `include_usage` appends "
            "has an empty `choices` list."
        )
    )
    usage: Usage | None = Field(
        default=None, description="Present only on the final frame, with `include_usage` set."
    )


class CompletionChoice(_Passthrough):
    """One candidate completion from the legacy endpoint."""

    index: int = Field(description="Position of this choice in the response.", ge=0)
    text: str = Field(description="The generated text, including any `echo` prefix.")
    finish_reason: str | None = Field(
        default=None,
        description="Why generation stopped: `stop`, `length`, or a provider-specific value.",
    )
    logprobs: dict[str, Any] | None = Field(
        default=None, description="Per-token log probabilities, when `logprobs` was requested."
    )


class CompletionResponse(_Passthrough):
    """Body for a non-streaming `POST /v1/completions`."""

    id: str = Field(description="Unique id for this completion, e.g. `cmpl-...`.")
    object: str = Field(default="text_completion", description="Always `text_completion`.")
    created: int = Field(description="Unix timestamp the completion was created at.", ge=0)
    model: str = Field(description="The model that actually served the call.")
    choices: list[CompletionChoice] = Field(
        description="One choice per prompt, in the order the prompts were sent."
    )
    usage: Usage | None = Field(default=None, description="Token counts, when the provider reports them.")


class Embedding(_Passthrough):
    """One vector."""

    object: str = Field(default="embedding", description="Always `embedding`.")
    index: int = Field(description="Position of this vector within the response.", ge=0)
    embedding: list[float] | str = Field(
        description=(
            "The vector itself: an array of floats, or a base64 string when "
            "`encoding_format=base64` was requested."
        )
    )


class EmbeddingResponse(_Passthrough):
    """Body for `POST /v1/embeddings`."""

    object: str = Field(default="list", description="Always `list`.")
    data: list[Embedding] = Field(
        description="One embedding per input, in the order the inputs were sent."
    )
    model: str = Field(description="The model that actually served the call.")
    usage: Usage | None = Field(
        default=None,
        description="Token counts. `completion_tokens` is 0 -- nothing was generated.",
    )


class ImageGenerationResponse(_Passthrough):
    """Body for `POST /v1/images/generations`."""

    created: int = Field(description="Unix timestamp the images were created at.", ge=0)
    data: list[dict[str, Any]] = Field(
        description=(
            "One entry per image, each holding `url` (default) or `b64_json`, and "
            "sometimes a `revised_prompt` when the provider rewrote the prompt."
        )
    )


class TranscriptionResponse(_Passthrough):
    """Body for `POST /v1/audio/transcriptions`.

    Shape follows `response_format`. `json` returns just `text`; `verbose_json`
    adds `language`, `duration`, `segments` and per-word `words`. The `text`,
    `srt` and `vtt` formats are not JSON at all -- they come back as a bare
    string under the matching media type, which the route's `responses` documents
    alongside this.
    """

    text: str = Field(description="The transcribed text.")


class ModelResponse(_Passthrough):
    """One entry in the model catalog returned by `GET /v1/models`."""

    id: str = Field(
        description=(
            "The model id to send as `model`. Provider-qualified as "
            "`<namespace>/<provider>/<model>` when `gateway.expose` is `prefixed`, "
            "bare when it is `raw`, and both are listed when it is `both`. Send "
            "back exactly the id you were given here."
        )
    )
    object: str = Field(default="model", description="Always `model`.")
    created: int = Field(description="Unix timestamp the entry was generated at.", ge=0)
    owned_by: str = Field(
        description="`margAI` for a virtual or dynamic entry; otherwise the provider that serves it."
    )
    parent: str | None = Field(
        default=None,
        description="The underlying model id, for a namespaced or virtual entry."
    )
    virtual: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Present only on a virtual-model entry: the policy deciding its target "
            "per call (strategy, eligible providers and models, exclusions, "
            "preferences, cost ceiling)."
        ),
    )


class ModelsResponse(_Passthrough):
    """Body for `GET /v1/models`."""

    object: str = Field(default="list", description="Always `list`.")
    data: list[ModelResponse] = Field(
        description=(
            "Every model this gateway accepts, in `<namespace>/<provider>/<model>` "
            "form. Always includes the reserved dynamic id, so a client picking "
            "from a dropdown can still reach the routing chain."
        )
    )


class GatewayDescriptionResponse(_Passthrough):
    """Body for `GET /v1/<namespace>/tags` -- what this gateway can do."""

    gateway: dict[str, Any] = Field(
        description=(
            "Identity and policy: `name`, `prefix`, `expose` (how model ids are "
            "listed), `dynamic_model`, `default_provider`, `on_unknown_tag`."
        )
    )
    tags: dict[str, list[dict[str, Any]]] = Field(
        description=(
            "Registered tags, keyed by namespace, then by qualified `id`. Read "
            "from the live registry, not from the config, so an entry here always "
            "has a handler behind it. Each entry carries `name`, the display "
            "`namespace`, the first line of the handler's docstring, and the "
            "phases it runs in."
        )
    )
    namespaces: list[str] = Field(
        description="Every namespace a tag can be qualified with, the first being the app's own.",
    )
    virtual_models: list[dict[str, Any]] = Field(
        description=(
            "Each `[models.*]` policy: its strategy, eligible providers and models, "
            "exclusions, preferences and cost ceiling."
        )
    )
    providers: list[dict[str, Any]] = Field(
        description="Each configured provider, its kind, base URL and the models it offers."
    )
    packs: list[dict[str, Any]] = Field(
        description=(
            "Each discovered pack, whether it installed, the tags it claimed, and "
            "the error if it did not."
        )
    )
    failures: list[dict[str, Any]] = Field(
        description="Packs that failed to install, flattened out of `packs` for the common case.",
    )
    config: dict[str, Any] = Field(
        description=(
            "Where the effective configuration came from: the file it was read "
            "from, the layers merged over it, and the per-pack tables in force."
        )
    )


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


#: The OpenAPI document's own description. `<namespace>` is substituted with the
#: live `gateway.prefix` in :func:`build_app`, so a gateway configured with a
#: different prefix documents itself correctly instead of lying about `margAI`.
APP_DESCRIPTION = """\
An OpenAI-compatible API in front of margAI. Point any OpenAI SDK at this
server and call it exactly as you would call OpenAI.

**Requests are OpenAI's, answers are yours.** Every OpenAI field is accepted
and forwarded verbatim. A routing chain runs first: `before` hooks can rewrite
the request, bangtag handlers activated by the request can act on it, and
`after` hooks can shape what comes back. The result is that the request you see
in your logs is the request you sent and the response you get is the response
the hooks produced -- so `model` in the response is the model that actually
served the call, which may not be the one you asked for.

**Model ids.** List them with `GET /v1/models` and send one back as `model`:
`<namespace>/<provider>/<model>` targets a provider explicitly, a bare
`<model>` resolves across configured providers, `<namespace>/dynamic` hands the
choice to the routing chain per call, and `<namespace>/<name>` selects a virtual
model whose policy picks a target per call. An id that does not resolve is a 404
naming the ids that do.

**Bangtags.** A leading `!<namespace>: <tag>` directive in the last user
message activates a tag handler and is stripped before the prompt reaches the
provider. `GET /v1/<namespace>/tags` lists what is registered. `<namespace>` is
this gateway's `gateway.prefix`.

**Errors.** Every non-2xx response is `{"error": {"message", "type", "param",
"code"}}`. A malformed body is `400`, an unresolvable model `404`, and an
upstream failure keeps the provider's own status and error type.
"""

OPENAPI_TAGS: list[dict[str, Any]] = [
    {
        "name": "Models",
        "description": "The model catalog: what this gateway will accept, and what it can do.",
    },
    {
        "name": "Chat",
        "description": "OpenAI chat completions, streaming and non-streaming. The primary surface.",
    },
    {
        "name": "Completions",
        "description": (
            "The legacy text-completion endpoint. No system message, and not every "
            "provider serves it -- prefer Chat for new work."
        ),
    },
    {
        "name": "Embeddings",
        "description": "Turns text into vectors. Streams nothing; `after` hooks can still shape the vectors.",
    },
    {
        "name": "Images",
        "description": "Image generation. The entire creative brief is `prompt`; the model sees nothing else.",
    },
    {
        "name": "Audio",
        "description": (
            "Audio transcription. The only multipart endpoint: `file` plus the "
            "optional fields as form parts, not a JSON body."
        ),
    },
]

#: Attached to every route that can fail, so a reader of `/docs` sees the failure
#: modes next to the success shape instead of discovering them in production.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": (
            "The request was rejected before it reached a provider: a body that is "
            "not valid JSON, a field that failed validation, an empty upload, or a "
            "configuration error. `error.param` names the offending field."
        ),
    },
    404: {
        "model": ErrorResponse,
        "description": "A model id was not found. Only `GET /v1/models/{model_id}` produces this.",
    },
    500: {
        "model": ErrorResponse,
        "description": (
            "An unhandled failure. `error.message` is deliberately generic; the "
            "detail is in the server log."
        ),
    },
}

#: FastAPI generates a 422 for every route with a validated parameter, but
#: margAI replaces that handler, so no request ever gets a 422. Overriding the
#: entry keeps the generated document honest instead of advertising a status the
#: server never returns.
_VALIDATION_NOTE: dict[int | str, dict[str, Any]] = {
    422: {
        "model": ErrorResponse,
        "description": (
            "Never returned. margAI replaces FastAPI's validation handler, so a "
            "malformed body or a field that fails validation comes back as a **400** "
            "with this same body shape -- see 400."
        ),
    },
}


def build_app(wrapper: Wrapper | None = None, config: Config | None = None) -> FastAPI:
    """Build a FastAPI application around a wrapper.

    Pass either an existing :class:`Wrapper` (tests / embedded use) or a
    :class:`Config` to construct one from configuration.

    The whole routed surface is documented through the models and route metadata
    above, which is what `/docs`, `/redoc` and `/openapi.json` render.
    """
    wrapper = wrapper or Wrapper.from_config(config or load_config())
    prefix = wrapper.router.prefix

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await wrapper.aclose()

    from .. import __version__

    app = FastAPI(
        title=f"{wrapper.name} gateway",
        version=__version__,
        description=APP_DESCRIPTION.replace("<namespace>", prefix),
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(400, _validation_message(exc), "invalid_request_error")

    # -------------------------------------------------------------------------
    # Root
    # -------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {"name": wrapper.name, "status": "ok", "prefix": prefix}

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    @app.get(
        "/v1/models",
        response_model=ModelsResponse,
        tags=["Models"],
        summary="List the models this gateway accepts",
        description=(
            "The catalog, taken from the live router rather than from the config, "
            "so an entry listed here is a model that will actually resolve. Each "
            "id is what you send back as `model`."
        ),
        responses=_ERROR_RESPONSES,
    )
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": await wrapper.models()}

    @app.get(
        "/v1/models/{model_id}",
        response_model=ModelResponse,
        tags=["Models"],
        summary="Retrieve one model from the catalog",
        description=(
            "Look up a single entry by the exact id `GET /v1/models` returned. The "
            "id is compared literally, so an unprefixed spelling is a 404 when the "
            "catalog is prefixed."
        ),
        responses={**_ERROR_RESPONSES, **_VALIDATION_NOTE},
    )
    async def retrieve_model(model_id: str) -> dict[str, Any] | JSONResponse:
        for model in await wrapper.models():
            if model.get("id") == model_id:
                return model
        return _error_response(404, f"Model '{model_id}' not found", "not_found_error")

    # -------------------------------------------------------------------------
    # Tag introspection
    # -------------------------------------------------------------------------
    @app.get(
        f"/v1/{prefix}/tags",
        response_model=GatewayDescriptionResponse,
        tags=["Models"],
        summary="Describe this gateway: its config and its registered tags",
        description=(
            "Introspection, not routing -- it answers no LLM call. `tags` is read "
            "from the live registry, so every entry listed has a handler behind it; "
            "the human-readable summary next to each is the first line of that "
            "handler's docstring. The path is namespaced by `gateway.prefix`, so it "
            f"is `GET /v1/{prefix}/tags` on this gateway."
        ),
        responses=_ERROR_RESPONSES,
    )
    async def describe_gateway() -> dict[str, Any]:
        return wrapper.describe()

    # -------------------------------------------------------------------------
    # Chat Completions
    # -------------------------------------------------------------------------
    @app.post(
        "/v1/chat/completions",
        response_model=None,
        tags=["Chat"],
        summary="Create a chat completion",
        description=(
            "The primary endpoint. With `stream` false you get one JSON body; with "
            "`stream` true the same envelope arrives as server-sent events, each "
            "frame carrying a `delta`, terminated by `data: [DONE]`. Add "
            "`stream_options: {\"include_usage\": true}` and the stream ends with a "
            "usage-only frame, so a streamed call can be costed the same way a "
            "non-streamed one is."
        ),
        responses={
            200: {
                "content": {
                    "application/json": {
                        "schema": ChatCompletionResponse.model_json_schema(),
                    },
                    "text/event-stream": {
                        "schema": ChatCompletionStreamChunk.model_json_schema(),
                        "example": (
                            'data: {"id": "chatcmpl-1", "object": "chat.completion.chunk", '
                            '"choices": [{"index": 0, "delta": {"content": "Hi"}}]}\n\n'
                        ),
                    },
                },
                "description": (
                    "JSON when `stream` is false; server-sent events when it is true. "
                    "In the JSON case `usage` is absent when the provider omits it, "
                    "and `message.content` is `null` when the model is calling a tool "
                    "instead of answering -- read `tool_calls` before assuming there "
                    "is nothing to show the user."
                ),
            },
            **_ERROR_RESPONSES,
            **_ERROR_RESPONSES,
            **_VALIDATION_NOTE,
        },
    )
    async def chat_completions(body: ChatCompletionRequest = Body(...)) -> JSONResponse | StreamingResponse:
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="chat")

    # -------------------------------------------------------------------------
    # Completions (Legacy)
    # -------------------------------------------------------------------------
    @app.post(
        "/v1/completions",
        response_model=None,
        tags=["Completions"],
        summary="Create a legacy text completion",
        description=(
            "Prompt-in, text-out, with no notion of a system message. Prefer "
            "`/v1/chat/completions` for new work -- this endpoint is not served by "
            "every provider."
        ),
        responses={
            200: {
                "content": {
                    "application/json": {
                        "schema": CompletionResponse.model_json_schema(),
                    },
                    "text/event-stream": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "object": {"type": "string", "const": "text_completion"},
                                "created": {"type": "integer"},
                                "model": {"type": "string"},
                                "choices": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {"type": "integer"},
                                            "text": {"type": "string"},
                                            "finish_reason": {"type": ["string", "null"]},
                                        },
                                    },
                                },
                                "usage": {"type": ["object", "null"]},
                            },
                        },
                        "example": (
                            'data: {"id": "cmpl-1", "object": "text_completion", '
                            '"choices": [{"index": 0, "text": " world"}]}\n\n'
                        ),
                    },
                },
                "description": "The generated text, one choice per prompt, in the order sent.",
            },
            **_ERROR_RESPONSES,
            **_ERROR_RESPONSES,
            **_VALIDATION_NOTE,
        },
    )
    async def completions(body: CompletionRequest = Body(...)) -> JSONResponse | StreamingResponse:
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="completions")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    @app.post(
        "/v1/embeddings",
        response_model=None,
        tags=["Embeddings"],
        summary="Embed text into vectors",
        description=(
            "One vector per input, returned in the order the inputs were sent. "
            "Never streams. `after` hooks can still shape the vectors, so the "
            "output is the gateway's rather than necessarily the provider's."
        ),
        responses={
            200: {
                "model": EmbeddingResponse,
                "description": (
                    "The vectors. `embedding` is an array of floats, or a base64 "
                    "string when `encoding_format=base64` was requested. "
                    "`usage.completion_tokens` is 0: nothing was generated."
                ),
            },
            **_ERROR_RESPONSES,
            **_ERROR_RESPONSES,
            **_VALIDATION_NOTE,
        },
    )
    async def embeddings(body: EmbeddingRequest = Body(...)) -> JSONResponse | StreamingResponse:
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="embeddings")

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------
    @app.post(
        "/v1/images/generations",
        response_model=None,
        tags=["Images"],
        summary="Generate images from a prompt",
        description=(
            "`prompt` is the whole brief -- the model is given no other context, so "
            "the words that matter are the words that go here."
        ),
        responses={
            200: {
                "model": ImageGenerationResponse,
                "description": (
                    "One entry per image, holding `url` or `b64_json`. A `url` "
                    "expires, so download it rather than storing it; `b64_json` "
                    "avoids that at the cost of a much larger body."
                ),
            },
            **_ERROR_RESPONSES,
            **_ERROR_RESPONSES,
            **_VALIDATION_NOTE,
        },
    )
    async def images_generations(
        body: ImageGenerationRequest = Body(...),
    ) -> JSONResponse | StreamingResponse:
        return await _forward(body.model_dump(exclude_none=True), wrapper, kind="images_generations")

    # -------------------------------------------------------------------------
    # Audio (multipart: the only genuinely multipart endpoint we route)
    # -------------------------------------------------------------------------
    @app.post(
        "/v1/audio/transcriptions",
        response_model=None,
        tags=["Audio"],
        summary="Transcribe an audio file",
        description=(
            "Sent as `multipart/form-data`, not JSON: the `file` part plus whichever "
            "of the optional fields you need. Every field is documented individually "
            "below. `language` is worth sending -- it improves accuracy and saves the "
            "model the detection work. Note that only the structured response formats "
            "carry `usage`, so `text` and `srt` responses cannot be costed from the "
            "body alone."
        ),
        responses={
            200: {
                "model": TranscriptionResponse,
                "content": {
                    "application/json": {"schema": TranscriptionResponse.model_json_schema()},
                    "text/plain": {
                        "schema": {"type": "string"},
                        "example": "The transcribed text, bare.",
                    },
                    "application/x-subrip": {
                        "schema": {"type": "string"},
                        "example": "1\n00:00:00,000 --> 00:00:02,500\nThe transcribed text.\n",
                    },
                    "text/vtt": {
                        "schema": {"type": "string"},
                        "example": "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nThe transcribed text.\n",
                    },
                },
                "description": (
                    "The transcript. JSON for `json` and `verbose_json` (the latter "
                    "adding segments, language and timings); a bare string for `text`; "
                    "subtitle text for `srt` and `vtt`."
                ),
            },
            **_ERROR_RESPONSES,
            **_ERROR_RESPONSES,
            **_VALIDATION_NOTE,
        },
    )
    async def audio_transcriptions(
        file: UploadFile = File(
            ...,
            description=(
                "The audio to transcribe. Sent as a file part, not inline. Any "
                "container and sample rate the provider accepts; an empty file is "
                "rejected with a 400."
            ),
        ),
        model: str = Body(..., description=_MODEL_FIELD_HELP),
        language: str | None = Body(
            default=None,
            description=(
                "ISO-639-1 code for the audio, e.g. `en` or `fr`. Supplying it "
                "improves accuracy and avoids spending detection budget; omit it and "
                "the model detects the language itself."
            ),
        ),
        prompt: str | None = Body(
            default=None,
            description=(
                "Optional context to steer the transcription -- domain vocabulary, "
                "speaker names, or an earlier fragment. Not a general instruction."
            ),
        ),
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = Body(
            default="json",
            description=(
                "`json` returns `{\"text\": ...}`. `verbose_json` adds `segments`, "
                "`language` and per-word timings. `text` returns a bare string; `srt` "
                "and `vtt` return subtitle formats."
            ),
        ),
        temperature: float | None = Body(
            default=None,
            ge=0.0,
            le=1.0,
            description="Sampling temperature, 0 to 1. 0 is the most literal transcription.",
        ),
        timestamp_granularities: str | None = Body(
            default=None,
            description=(
                "Which timestamp levels to include in a `verbose_json` response, as a "
                "comma-separated list of `word` and/or `segment`. Requires "
                "`response_format=verbose_json`."
            ),
        ),
    ) -> JSONResponse | StreamingResponse:
        content = await file.read()
        if not content:
            return _error_response(400, "Uploaded audio file is empty", "invalid_request_error")
        body: dict[str, Any] = {
            "model": model,
            "file": (file.filename or "audio", content, file.content_type or "application/octet-stream"),
        }
        body.update(
            {
                key: value
                for key, value in (
                    ("language", language),
                    ("prompt", prompt),
                    ("response_format", response_format),
                    ("temperature", temperature),
                    ("timestamp_granularities", timestamp_granularities),
                )
                if value is not None
            }
        )
        return await _forward(body, wrapper, kind="audio_transcriptions")

    # -------------------------------------------------------------------------
    # Internal forward helper
    # -------------------------------------------------------------------------
    async def _forward(body: dict, wrapper: Wrapper, *, kind: str) -> JSONResponse | StreamingResponse:
        if body.get("stream", False) and kind in Wrapper._STREAMABLE_KINDS:
            try:
                handle = await wrapper.open_stream(body, kind=kind)
            except ApiError as exc:
                return JSONResponse(content=exc.body, status_code=exc.status)
            except Exception:  # pragma: no cover - defensive
                return _error_response(500, "Internal error", "server_error")
            return StreamingResponse(
                handle.lines(), media_type="text/event-stream", headers=ERROR_HEADERS
            )

        response = await wrapper.complete(body, kind=kind)
        return JSONResponse(content=response.body, status_code=response.status)

    return app
