"""Response types for tag functions to control pipeline flow.

Tag functions can return these special types to:
- Override routing: return RouteOverride("provider/model")
- Short-circuit: return ImmediateResponse({...})  
- Return error: return ErrorResponse(400, "message")

They share the :class:`ControlSignal` base so the pipeline can tell a return
value that means "do something" apart from one that means "here is a rewritten
request" -- a `before` handler returning a dict replaces the body, and treating
one of these as a body would put a frozen dataclass where a request was
expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ControlSignal", "ErrorResponse", "ImmediateResponse", "RouteOverride"]


class ControlSignal:
    """Base for the three values that redirect a call.

    Not a dataclass and never instantiated: it exists so the pipeline can ask
    ``isinstance(result, ControlSignal)`` once, rather than testing three types
    at each point a handler's return value is examined.
    """


@dataclass(frozen=True)
class RouteOverride(ControlSignal):
    """Override model routing from a tag function.
    
    Return this from a tag function to force routing to a specific model.
    `model` is a model id in the form ``GET /v1/models`` lists them, so
    ``margAI/local/llama3.2-1b`` under a prefixed gateway.
    
    Example::
    
        @app.tag("fast")
        def fast_route(ctx):
            return RouteOverride("margAI/local/llama3.2-1b")
    
    Or use the context method::
    
        @app.tag("fast")
        def fast_route(ctx):
            return ctx.route_to("margAI/local/llama3.2-1b")
    """
    
    model: str
    
    def __repr__(self) -> str:
        return f"RouteOverride({self.model!r})"


@dataclass(frozen=True)
class ImmediateResponse(ControlSignal):
    """Skip LLM and return this response immediately.
    
    Return this from a tag function to short-circuit the pipeline and
    return a cached or pre-computed response.
    
    Example::
    
        @app.tag("cache")
        def check_cache(ctx):
            key = ctx.cache_key()
            if cached := cache.get(key):
                return ImmediateResponse(cached)
            # No return = continue to LLM
    
    Or use the context method::
    
        @app.tag("cache")
        def check_cache(ctx):
            if cached := cache.get(ctx.cache_key()):
                return ctx.respond(cached)
    """
    
    payload: dict[str, Any]
    
    def __repr__(self) -> str:
        return f"ImmediateResponse(<{len(self.payload)} keys>)"


@dataclass(frozen=True)
class ErrorResponse(ControlSignal):
    """Return an error response immediately.
    
    Return this from a tag function to fail the request with an error.
    
    Example::
    
        @app.tag("validate")
        def validate(ctx):
            if len(ctx.user_message()) > 10000:
                return ErrorResponse(400, "Message too long")
    
    Or use the context method::
    
        @app.tag("validate")
        def validate(ctx):
            if len(ctx.user_message()) > 10000:
                return ctx.error(400, "Message too long")
    """
    
    status: int
    message: str
    error_type: str = "invalid_request_error"
    
    def __repr__(self) -> str:
        return f"ErrorResponse({self.status}, {self.message!r})"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI error format."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": None,
                "code": None,
            }
        }
