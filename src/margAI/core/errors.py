"""Shared exception type and streaming sentinels."""

from __future__ import annotations

from typing import Any

__all__ = ["DONE", "ApiError"]


class _Done:
    """Sentinel returned by providers when the upstream stream ends."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<DONE>"


DONE = _Done()


class ApiError(Exception):
    """An error that maps to a JSON (OpenAI-compatible) error body."""

    def __init__(
        self,
        status: int = 500,
        message: str = "Internal server error",
        error_type: str = "server_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

    @property
    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

    @classmethod
    def from_openai_body(cls, body: Any, implicit_status: int = 500) -> "ApiError":
        """Best-effort parse of an upstream OpenAI-style error payload."""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                # Some providers (Anthropic) nest further: error.error.{...}
                if isinstance(err.get("error"), dict):
                    err = err["error"]
                return cls(
                    status=int(err.get("status") or implicit_status),
                    message=str(err.get("message") or "Upstream error"),
                    error_type=str(err.get("type") or "upstream_error"),
                    param=err.get("param"),
                    code=err.get("code"),
                )
            message = str(body.get("message") or body)
        else:
            message = str(body) if body is not None else "Upstream error"
        return cls(status=implicit_status, message=message, error_type="upstream_error")