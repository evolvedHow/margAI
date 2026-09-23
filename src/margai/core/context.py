"""Per-call context passed to interceptors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RequestContext"]


@dataclass
class RequestContext:
    """Mutable, per-request state carried through the wrapper pipeline.

    - ``body``: the OpenAI-shaped request body (``messages``, ``model``,
      sampling params...). Hooks may mutate it in place or return a
      replacement dict from a ``request`` hook.
    - ``state``: arbitrary scratch space private to one call. Use this to
      thread data from a ``request`` hook to a ``response``/``stream`` hook
      (this is the ``ctx.state`` pattern from the original margai sketch).
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

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.body.get("messages", [])

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