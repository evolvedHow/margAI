"""Per-call context passed to interceptors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .intent import INTENT_KEY, RoutingIntent
from .marglets import ACTIVE_KEY

__all__ = ["RequestContext"]


@dataclass
class RequestContext:
    """Mutable, per-request state carried through the wrapper pipeline.

    - ``body``: the OpenAI-shaped request body (``messages``, ``model``,
      sampling params...). Hooks may mutate it in place or return a
      replacement dict from a ``request`` hook.
    - ``state``: arbitrary scratch space private to one call. Use this to
      thread data from a ``request`` hook to a ``response``/``stream`` hook
      (this is the ``ctx.state`` pattern from the original margAI sketch).
    - ``provider`` / ``upstream_model`` / ``request_model``: populated by the
      router before transport is reached.
    """

    body: dict[str, Any]
    stream: bool = False
    provider: str | None = None
    upstream_model: str | None = None
    request_model: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    config: Any = None
    started_at: float = field(default_factory=time.perf_counter)

    # -- tags and marglets --------------------------------------------------

    @property
    def tags(self) -> list[Any]:
        """The bangtags carried by this call (empty when the layer is off)."""
        return self.state.get("bangtags") or []

    @property
    def marglets(self) -> list[str]:
        """Names of the marglets this call activated, in dispatch order."""
        return list(self.state.get(ACTIVE_KEY) or [])

    @property
    def intent(self) -> RoutingIntent:
        """This call's routing intent, created on first access from its tags.

        The intent is immutable, so a marglet refines it with
        :meth:`steer` (``ctx.steer(provider="local")``) or by assigning
        ``ctx.intent = ctx.intent.with_(...)``. The dynamic router reads
        whatever is here by the time routing happens.
        """
        intent = self.state.get(INTENT_KEY)
        if intent is None:
            intent = RoutingIntent.from_tags(self.tags)
            self.state[INTENT_KEY] = intent
        return intent

    @intent.setter
    def intent(self, value: RoutingIntent) -> None:
        self.state[INTENT_KEY] = value

    def steer(self, **changes: Any) -> RoutingIntent:
        """Refine the intent and return it: ``ctx.steer(provider="local")``."""
        self.intent = self.intent.with_(**changes)
        return self.intent

    @property
    def dynamic(self) -> bool:
        """True when the client asked for the reserved dynamic model id."""
        return self.state.get("marglets.dynamic", False)

    # -- body helpers -------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.body.get("messages", [])

    @property
    def prompt(self) -> Any:
        """The legacy-completions prompt (``/v1/completions``); None on chat calls."""
        return self.body.get("prompt")

    def add_system_prompt(self, text: str, *, prepend: bool = True) -> None:
        """Insert or merge a system message.

        If a system message already exists its content is merged with
        ``text`` (prepended by default, since stable instructions belong
        early for prefix caching).
        """
        msgs = self.body.setdefault("messages", [])
        for msg in msgs:
            if msg.get("role") == "system":
                existing = str(msg.get("content", ""))
                msg["content"] = (
                    (text + "\n\n" + existing) if prepend else (existing + "\n\n" + text)
                )
                return
        msgs.insert(0, {"role": "system", "content": text})

    def last_user_message(self) -> str | None:
        for msg in reversed(self.messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)
