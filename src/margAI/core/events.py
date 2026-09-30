"""Named event decorators: the exit points around every call.

An :class:`EventRegistry` holds *tag handlers* keyed by name across the key
events in a call's life:

- ``before`` -- runs on ``ctx`` before the request goes upstream; may mutate
  ``ctx.body`` / ``ctx.state`` or raise.
- ``after``  -- runs on the non-streaming ``payload`` dict; return a payload.
- ``stream`` -- runs per SSE chunk on streaming calls; return the chunk (or
  ``None`` to drop it). Handlers share a per-call mutable scratch dict.
- ``error`` -- shapes failures; return a dict to take over the error body.

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

:func:`~margAI.Wrapper.on` is the uniform entry point over the same registry,
spelling the phase as an argument and carrying a namespace and an order.

Storage, ordering and pack ownership live in
:class:`~margAI.core.tags.TagRegistry`; this class owns only the calling
convention, which is what differs per phase.

You usually never touch an :class:`EventRegistry` directly: the
:class:`~margAI.Wrapper` owns one (``wrapper.events``) and its decorators
wire events automatically, dispatching on tags read from
``ctx.state["bangtags"]``. See :class:`~margAI.Wrapper` for the public
surface.

Tags are duck-typed -- anything carrying a ``.name`` works, and a ``.namespace``
is optional. Handlers may be sync or async.
"""

from __future__ import annotations

import logging
from typing import Any

from .hooks import maybe_await
from .marglets import PHASES
from .responses import ControlSignal
from .tags import (
    RESERVED_NAMESPACE,
    ROOT_PACK,
    TagConflictError,
    TagHandler,
    TagRegistry,
    qualified_key,
)

__all__ = [
    "EVENTS",
    "EventHandler",
    "EventRegistry",
    "TagConflictError",
    "TagHandler",
    "TagRegistry",
]

logger = logging.getLogger("margAI.events")

# The four phase names, aliased rather than re-spelled. ``marglets.PHASES`` is
# the definition; this is the name this module has always exported.
EVENTS = PHASES


class EventRegistry:
    """Tag-keyed handlers across the before/after/stream/error events."""

    def __init__(self, namespace: str | None = None) -> None:
        self.tags = TagRegistry(namespace) if namespace else TagRegistry()

    # -- registration -------------------------------------------------------

    def on(
        self,
        name: str,
        *,
        phase: str = "before",
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        """Register ``name`` in ``phase``. The uniform entry point."""
        return self._put(name, phase, order=order, replace=replace, namespace=namespace, pack=pack)

    def before(
        self,
        name: str,
        *,
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        """Register a request-phase handler: ``fn(ctx, tag)``."""
        return self._put(name, "before", order=order, replace=replace, namespace=namespace, pack=pack)

    def after(
        self,
        name: str,
        *,
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        """Register a non-streaming response-phase handler: ``fn(payload, ctx,
        tag) -> payload``."""
        return self._put(name, "after", order=order, replace=replace, namespace=namespace, pack=pack)

    def stream(
        self,
        name: str,
        *,
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        """Register a per-chunk handler: ``fn(chunk, ctx, tag, scratch)``."""
        return self._put(name, "stream", order=order, replace=replace, namespace=namespace, pack=pack)

    def error(
        self,
        name: str,
        *,
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        """Register a failure-phase handler: ``fn(exc, ctx, tag) -> dict |
        None`` (return a dict to take over the error body)."""
        return self._put(name, "error", order=order, replace=replace, namespace=namespace, pack=pack)

    def _put(
        self,
        name: str,
        phase: str,
        *,
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
        pack: str = ROOT_PACK,
    ) -> Any:
        if phase not in EVENTS:
            raise ValueError(f"unknown phase {phase!r}; expected one of {', '.join(EVENTS)}")

        def deco(fn: Any) -> Any:
            existing = self.tags.handlers(qualified_key(name, namespace or RESERVED_NAMESPACE), phase)
            if existing and existing[0].fn is not fn and not replace:
                logger.warning(
                    "replacing %s handler for %r (was %s, now %s); one name means one handler",
                    phase,
                    name,
                    getattr(existing[0].fn, "__qualname__", existing[0].fn),
                    getattr(fn, "__qualname__", fn),
                )
            self.tags.register(
                name=name,
                namespace=namespace or RESERVED_NAMESPACE,
                phase=phase,
                fn=fn,
                pack=pack,
                order=order,
                replace=replace,
            )
            return fn

        return deco

    # -- base-class support ------------------------------------------------

    def add(self, handler: Any, *, namespace: str | None = None, pack: str = ROOT_PACK) -> EventRegistry:
        """Register every ``{event}_{name}`` method found on ``handler``
        (an instance -- methods are bound; a callable class works if its
        events are staticmethods)."""
        for event in EVENTS:
            prefix = f"{event}_"
            for attr in dir(handler):
                if not attr.startswith(prefix):
                    continue
                name = attr[len(prefix) :]
                if not name:
                    continue
                self.tags.register(
                    name=name,
                    namespace=namespace or RESERVED_NAMESPACE,
                    phase=event,
                    fn=getattr(handler, attr),
                    pack=pack,
                )
        return self

    # -- lookup ------------------------------------------------------------

    def all(self) -> set[str]:
        """Unqualified tag names with at least one handler."""
        return self.tags.bare_names()

    def describe(self) -> dict[str, str]:
        """Back-compat view: bare tag name -> first doc line."""
        return self.tags.docs()

    def __len__(self) -> int:
        return len(self.tags)

    # -- dispatch ----------------------------------------------------------

    async def run_before(self, tags: list[Any], ctx: Any) -> Any:
        """Run the ``before`` handlers for these tags.

        Returns the first :class:`~margAI.core.responses.ControlSignal` a handler
        returned -- ``ctx.route_to()``, ``ctx.respond()`` or ``ctx.error()`` --
        and stops there, because a signal is a decision about the whole call and
        running the handlers after it would act on a call that is already
        answered. The wrapper acts on the signal; a bare ``None`` means carry on.
        """
        for handler, tag in self.tags.plan(tags, "before"):
            out = await maybe_await(handler.fn(ctx, tag))
            if isinstance(out, ControlSignal):
                return out
        return None

    async def run_after(self, tags: list[Any], payload: dict, ctx: Any) -> dict:
        for handler, tag in self.tags.plan(tags, "after"):
            out = await maybe_await(handler.fn(payload, ctx, tag))
            if out is not None:
                payload = out
        return payload

    async def run_stream(self, tags: list[Any], chunk: dict, ctx: Any, scratch: dict) -> dict | None:
        for handler, tag in self.tags.plan(tags, "stream"):
            out = await maybe_await(handler.fn(chunk, ctx, tag, scratch))
            if out is None:
                return None
            chunk = out
        return chunk

    async def run_error(self, tags: list[Any], exc: Exception, ctx: Any) -> dict | None:
        for handler, tag in self.tags.plan(tags, "error"):
            out = await maybe_await(handler.fn(exc, ctx, tag))
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
