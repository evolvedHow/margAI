"""Shared exception type and streaming sentinels."""

from __future__ import annotations

from enum import Enum
from typing import Any, Final

__all__ = ["DONE", "ApiError", "DoneSentinel"]


class DoneSentinel(Enum):
    """The value providers return to signal end-of-stream.

    An enum rather than a plain sentinel object so that type checkers can
    narrow on `x is DONE` -- with an opaque object they cannot, and every
    `parse_chunk` implementation would need a cast to pass its result on.
    """

    TOKEN = "done"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<DONE>"

    def __str__(self) -> str:
        # Matches __repr__, so f-strings and logs don't disagree with each other.
        return "<DONE>"


DONE: Final[DoneSentinel] = DoneSentinel.TOKEN


class ApiError(Exception):
    """An error that maps to a JSON (OpenAI-compatible) error body.

    ``ctx`` is attached by the wrapper when the failure happens after a
    context exists, so error hooks and telemetry can see which call failed
    (which model it asked for, which marglets it activated) even when the
    failure was raised before the upstream was ever reached.
    """

    def __init__(
        self,
        status: int = 500,
        message: str = "Internal server error",
        error_type: str = "server_error",
        param: str | None = None,
        code: str | None = None,
        *,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        self._body = body
        self.ctx: Any = None

    @property
    def body(self) -> dict[str, Any]:
        if self._body is not None:
            return self._body
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

    def with_body(self, body: dict[str, Any]) -> ApiError:
        """A copy carrying ``body`` -- how an ``error`` hook takes over the
        response, including on the streaming path where the error is raised
        instead of returned."""
        clone = ApiError(self.status, self.message, self.error_type, self.param, self.code, body=body)
        clone.ctx = self.ctx
        return clone

    @classmethod
    def from_openai_body(cls, body: Any, implicit_status: int = 500) -> ApiError:
        """Best-effort parse of an upstream OpenAI-style error payload.

        Nothing in here may raise: it runs on the failure path, so a parser
        that blew up would turn an upstream 4xx into an unhandled 500 and lose
        the very message it was trying to recover. Unknown or malformed fields
        fall back rather than fail.
        """
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                # Anthropic nests once: {"type": "error", "error": {...}},
                # which this already handles. A few gateways double-wrap it, so
                # unwrap one more level if that is what we are looking at.
                if isinstance(err.get("error"), dict):
                    err = err["error"]
                return cls(
                    status=_as_status(err.get("status"), implicit_status),
                    message=str(err.get("message") or "Upstream error"),
                    error_type=str(err.get("type") or "upstream_error"),
                    param=err.get("param"),
                    code=err.get("code"),
                )
            message = str(body.get("message") or body)
        else:
            message = str(body) if body is not None else "Upstream error"
        return cls(status=implicit_status, message=message, error_type="upstream_error")


def _as_status(value: Any, fallback: int) -> int:
    """An int status out of an untrusted upstream field, or ``fallback``.

    A non-numeric ``error.status`` must not raise out of :meth:`from_openai_body`
    -- that is the error path, and an ``int()`` blowing up there discards the
    real message for a generic 500.
    """
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return fallback
    return fallback
