"""Anthropic provider (``kind = "anthropic"``).

Translates between the framework's internal OpenAI-compatible shape and
Anthropic's ``/v1/messages`` surface:

- ``prepare_chat`` folds ``system``/``developer`` messages into the
  ``system`` field, maps ``stop`` to ``stop_sequences``, and requires
  ``max_tokens`` (a default is taken from config ``extra.max_tokens``).
- ``parse_response`` / ``parse_chunk`` map Anthropic messages back to the
  OpenAI chat shape hooks and clients expect. Streaming usage (input/output
  tokens) is harvested from the ``message_delta`` event so telemetry works.
"""

from __future__ import annotations

import time
from typing import Any

from ..core import DONE
from ..core.errors import ApiError
from ..core.protocol import PreparedRequest, Transport, UpstreamResponse
from .openai_compat import Provider, _parse_sse_data_line

__all__ = ["AnthropicProvider"]

ANTHROPIC_VERSION = "2023-06-01"

_ROLES = {"user", "assistant"}
_SYSTEM_ROLES = {"system", "developer"}


class AnthropicProvider(Provider):
    """Anything that speaks Anthropic's ``/v1/messages`` API."""

    def headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        } if json_body else {"anthropic-version": ANTHROPIC_VERSION}
        extra = self.config.extra
        if isinstance(extra, dict) and isinstance(extra.get("headers"), dict):
            headers.update({str(k): str(v) for k, v in extra["headers"].items()})
        key = self.config.api_key
        if key:
            headers["x-api-key"] = key
        return headers

    def prepare_chat(self, ctx: Any) -> PreparedRequest:
        body = _anthropic_request_body(ctx, max_tokens=_default_max_tokens(self.config))
        return PreparedRequest(
            method="POST",
            url=self.endpoint("messages"),
            headers=self.headers(),
            json=body,
            timeout=self.config.timeout,
        )

    def parse_response(self, resp: UpstreamResponse, ctx: Any) -> tuple[dict, int]:
        status = resp.status
        if status >= 400:
            raise ApiError.from_openai_body(resp.body, implicit_status=status)
        return _anthropic_to_chat(resp.body, ctx.upstream_model), status

    def parse_chunk(self, raw_line: str, ctx: Any) -> dict | None:
        data = _parse_sse_data_line(raw_line)
        if data is DONE:
            return DONE
        if data is None:
            return None
        return _anthropic_event_to_chunk(data, ctx.upstream_model)

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
            raise ApiError.from_openai_body(resp.body, implicit_status=resp.status)
        data = resp.body.get("data", []) if isinstance(resp.body, dict) else []
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]


# -- translation helpers ------------------------------------------------------


def _anthropic_request_body(ctx: Any, *, max_tokens: int) -> dict[str, Any]:
    body = ctx.body if isinstance(ctx.body, dict) else {}
    messages, system = _split_messages(body.get("messages") or [])
    out: dict[str, Any] = {
        "model": ctx.upstream_model or body.get("model"),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        out["system"] = system
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop is not None:
        out["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    if body.get("user"):
        out["metadata"] = {"user_id": str(body["user"])}
    if ctx.stream:
        out["stream"] = True
    return out


def _split_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in _SYSTEM_ROLES:
            system_parts.append(_content_text(content))
            continue
        if role not in _ROLES:
            continue
        text = _content_text(content)
        if text:
            out.append({"role": role, "content": text})
    if not out:  # Anthropic requires at least one user message
        out.append({"role": "user", "content": ""})
    elif out[0]["role"] != "user":  # and the first message must be a user turn
        out.insert(0, {"role": "user", "content": ""})
    return out, "\n\n".join(p for p in system_parts if p)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _default_max_tokens(config: Any) -> int:
    extra = getattr(config, "extra", None) if config is not None else None
    if isinstance(extra, dict):
        try:
            return int(extra.get("max_tokens", 1024))
        except (TypeError, ValueError):
            pass
    return 1024


def _anthropic_to_chat(body: Any, model: str | None) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"object": "chat.completion", "choices": []}
    text = _content_text(body.get("content") or [])
    usage = body.get("usage") or {}
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    return {
        "id": body.get("id") or "cmpl-anthropic",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or body.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _stop_reason(body.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": (prompt or 0) + (completion or 0),
        },
    }


def _anthropic_event_to_chunk(data: dict, model: str | None) -> dict | None:
    etype = data.get("type")
    chunk: dict[str, Any] = {
        "id": data.get("id") or "cmpl-anthropic",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    if etype == "content_block_delta":
        delta = data.get("delta") or {}
        if delta.get("type") == "text_delta":
            chunk["choices"][0]["delta"] = {"content": str(delta.get("text", ""))}
            return chunk
        return None  # non-text deltas (tool input JSON etc.) not modeled
    if etype == "message_delta":
        usage = data.get("usage") or {}
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        chunk["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": (prompt or 0) + (completion or 0),
        }
        chunk["choices"][0]["finish_reason"] = _stop_reason(
            (data.get("delta") or {}).get("stop_reason")
        )
        return chunk
    # message_start / content_block_start / content_block_stop / message_stop / ping
    return None


def _stop_reason(reason: str | None) -> str | None:
    if reason in ("end_turn", "stop_sequence"):
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    return reason or None