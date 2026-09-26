"""Named event decorators: the exit points around every call.

An :class:`EventRegistry` holds handlers keyed by name across the key events
in a call's life:

- ``before`` -- runs on ``ctx`` before the request goes upstream; may mutate
  ``ctx.body`` / ``ctx.state`` or raise.
- ``after``  -- runs on the non-streaming ``payload`` dict; return a payload.
- ``stream`` -- runs per SSE chunk on streaming calls; return the chunk (or
  ``None`` to drop it). Handlers share a per-call mutable scratch dict.
- ``error``  -- shapes failures; return a dict to take over the error body.

Register with a decorator that takes the name:

    registry = EventRegistry()

    @registry.before("route")
    def route_before(ctx, tag): ...

    @registry.after("uppercase")
    def uppercase_after(payload, ctx, tag):
        payload["choices"][0]["message"]["content"] = \
            payload["choices"][0]["message"]["content"].upper()
        return payload

or subclass :class:`EventHandler` and name methods ``{event}_{name}`` --
same dispatch, no decorator:

    class Handlers(EventHandler):
        def before_route(self, ctx, tag): ...
        def after_uppercase(self, payload, ctx, tag): ...

    registry = EventRegistry()
    registry.add(Handlers())

You usually never touch an :class:`EventRegistry` directly: the
:class:`~margAI.Wrapper` owns one (``wrapper.events``) and its decorators
wire events automatically, dispatching on tags read from
``ctx.state["bangtags"]``. See :class:`~margAI.Wrapper` for the public
surface.

Tags are duck-typed -- anything carrying a ``.name`` (and optionally a
``.value``) works. Handlers may be sync or async.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from .hooks import maybe_await

__all__ = ["EventHandler", "EventRegistry"]

logger = logging.getLogger("margAI.events")

EVENTS = ("before", "after", "stream", "error")

# Where the wrapper's event dispatch reads tags from, and where per-call
# stream handlers share scratch space.
STATE_KEY = "bangtags"
SCRATCH_KEY = "events.scratch"


class EventRegistry:
    """Name-keyed handlers across the before/after/stream/error events."""

    def __init__(self) -> None:
        self._handlers: dict[str, dict[str, Any]] = {event: {} for event in EVENTS}

    def _put(self, event: str, name: str, fn: Any) -> None:
        """Register a handler, warning on replacement.

        One name means one handler per event, so a second registration for the
        same name silently disables the first. That is almost always a
        duplicated name in a config or an import happening twice, and it
        presents as "my hook just stopped running" -- worth a warning.
        """
        existing = self._handlers[event].get(name)
        if existing is not None and existing is not fn:
            logger.warning(
                "replacing %s handler for %r (was %s, now %s); one name means one handler",
                event,
                name,
                getattr(existing, "__qualname__", existing),
                getattr(fn, "__qualname__", fn),
            )
        self._handlers[event][name] = fn

    # -- registration decorators ------------------------------------------

    def before(self, name: str):
        """Register a request-phase handler: ``fn(ctx, tag)``."""

        def deco(fn: Any) -> Any:
            self._put("before", name, fn)
            return fn

        return deco

    def after(self, name: str):
        """Register a non-streaming response-phase handler:
        ``fn(payload, ctx, tag) -> payload``."""

        def deco(fn: Any) -> Any:
            self._put("after", name, fn)
            return fn

        return deco

    def stream(self, name: str):
        """Register a per-chunk handler: ``fn(chunk, ctx, tag, scratch)``."""

        def deco(fn: Any) -> Any:
            self._put("stream", name, fn)
            return fn

        return deco

    def error(self, name: str):
        """Register a failure-phase handler:
        ``fn(exc, ctx, tag) -> dict | None`` (return a dict to take over the
        error body)."""

        def deco(fn: Any) -> Any:
            self._put("error", name, fn)
            return fn

        return deco

    # -- base-class support ------------------------------------------------

    def add(self, handler: Any) -> EventRegistry:
        """Register every ``{event}_{name}`` method found on ``handler``
        (an instance -- methods are bound; a callable class works if its
        events are staticmethods)."""
        for event in EVENTS:
            prefix = f"{event}_"
            for attr in dir(handler):
                if not attr.startswith(prefix):
                    continue
                name = attr[len(prefix):]
                if not name:
                    continue
                self._put(event, name, getattr(handler, attr))
        return self

    # -- lookup ------------------------------------------------------------

    def all(self) -> set[str]:
        known: set[str] = set()
        for handlers in self._handlers.values():
            known.update(handlers)
        return known

    def describe(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in sorted(self.all()):
            fn = next(
                (self._handlers[e][name] for e in EVENTS if name in self._handlers[e]),
                None,
            )
            doc = (inspect.getdoc(fn) or "").strip().splitlines()
            out[name] = doc[0] if doc else ""
        return out

    # -- dispatch ----------------------------------------------------------

    async def run_before(self, tags: list[Any], ctx: Any) -> None:
        for tag in tags:
            fn = self._handlers["before"].get(tag.name)
            if fn is not None:
                await maybe_await(fn(ctx, tag))

    async def run_after(self, tags: list[Any], payload: dict, ctx: Any) -> dict:
        for tag in tags:
            fn = self._handlers["after"].get(tag.name)
            if fn is None:
                continue
            out = await maybe_await(fn(payload, ctx, tag))
            if out is not None:
                payload = out
        return payload

    async def run_stream(self, tags: list[Any], chunk: dict, ctx: Any, scratch: dict) -> dict | None:
        for tag in tags:
            fn = self._handlers["stream"].get(tag.name)
            if fn is None:
                continue
            out = await maybe_await(fn(chunk, ctx, tag, scratch))
            if out is None:
                return None
            chunk = out
        return chunk

    async def run_error(self, tags: list[Any], exc: Exception, ctx: Any) -> dict | None:
        for tag in tags:
            fn = self._handlers["error"].get(tag.name)
            if fn is None:
                continue
            out = await maybe_await(fn(exc, ctx, tag))
            if out is not None:
                return out
        return None


class EventHandler:
    """Base class for handler groups.

    Subclass it and name methods ``{event}_{name}`` -- ``before_route``,
    ``after_uppercase``, ``stream_structure``, ``error_upstream`` -- then
    hand an instance to the wrapper:

        app.add_handler(MyHandlers())

    The name after ``{event}_`` becomes the tag name, exactly as if the
    method had been registered with ``@app.<event>("name")``.
    """