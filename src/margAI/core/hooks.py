"""Interceptor registry and the decorator surface.

Four hook kinds, two on each side of the upstream call:

- **request** hooks run *before* transport: ``hook(ctx)`` may mutate
  ``ctx.body`` / ``ctx.state`` or return a replacement body.
- **response** hooks run *after* a non-streaming call:
  ``hook(payload: dict, ctx) -> dict``.
- **stream** hooks run per SSE chunk on streaming calls:
  ``hook(chunk: dict, ctx) -> dict | None`` (return ``None`` to drop the
  chunk).
- **error** hooks shape failures: ``hook(exc, ctx) -> dict | None`` (return
  a dict to take over the error body).

Ordering: ``request`` hooks run in registration order; ``response`` /
``stream`` / ``error`` hooks run in **reverse** registration order, so the
outermost request mutation is unwound first (middleware-stack semantics).
An explicit ``order`` int shifts a hook relative to its peers (lower runs
first on the way in).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .context import RequestContext

__all__ = ["HookKind", "HookRegistry", "maybe_await"]

HookFn = Callable[..., Any]


class HookKind(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STREAM = "stream"
    ERROR = "error"


@dataclass
class Hook:
    kind: HookKind
    fn: HookFn
    name: str
    order: int = 0
    seq: int = 0


async def maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


class HookRegistry:
    """Ordered collection of registered hooks."""

    def __init__(self) -> None:
        self._hooks: list[Hook] = []

    def register(self, kind: HookKind, fn: HookFn, name: str, order: int = 0) -> Hook:
        hook = Hook(kind=kind, fn=fn, name=name, order=order, seq=len(self._hooks))
        self._hooks.append(hook)
        return hook

    def _sorted(self, kind: HookKind, *, reverse: bool) -> list[Hook]:
        hooks = [h for h in self._hooks if h.kind is kind]
        hooks.sort(key=lambda h: (h.order, h.seq), reverse=reverse)
        return hooks

    def requests(self) -> list[Hook]:
        return self._sorted(HookKind.REQUEST, reverse=False)

    def responses(self) -> list[Hook]:
        return self._sorted(HookKind.RESPONSE, reverse=True)

    def streams(self) -> list[Hook]:
        return self._sorted(HookKind.STREAM, reverse=True)

    def errors(self) -> list[Hook]:
        return self._sorted(HookKind.ERROR, reverse=True)

    def count(self, kind: HookKind | None = None) -> int:
        if kind is None:
            return len(self._hooks)
        return sum(1 for h in self._hooks if h.kind is kind)

    async def apply_response(self, payload: dict, ctx: RequestContext) -> dict:
        for hook in self.responses():
            out = await maybe_await(hook.fn(payload, ctx))
            if out is not None:
                payload = out
        return payload

    async def apply_stream(self, chunk: dict, ctx: RequestContext) -> dict | None:
        for hook in self.streams():
            out = await maybe_await(hook.fn(chunk, ctx))
            if out is None:
                return None
            chunk = out
        return chunk

    async def apply_error(self, exc: Exception, ctx: RequestContext) -> dict | None:
        for hook in self.errors():
            out = await maybe_await(hook.fn(exc, ctx))
            if out is not None:
                return out
        return None