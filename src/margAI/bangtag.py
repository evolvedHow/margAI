"""Bangtag directives -- an optional, separable layer of margAI.

A *bangtag* is a directive embedded in a user's message that changes how that
one call is handled, without touching configuration:

    "make it terse  !margAI: refine"
    "use the small model  !margAI: route=llama3.2:3b"
    "put the answer in bullets  !margAI: structure"

This module is just the layer. It provides:

- the parser: ``Tag`` / ``find_directive`` / ``remove_directive``; and
- ``install_bangtags(app)``, which wires one request hook that finds the
  directive, strips it from the prompt, and stores the tags in
  ``ctx.state["bangtags"]`` -- exactly where the wrapper's event dispatch
  looks (see :meth:`margAI.Wrapper.before`).

The layer is optional and stands apart from the event decorators: those work
with any tag source. If you don't want ``!margAI:`` syntax, skip
``install_bangtags`` and set ``ctx.state["bangtags"]`` from your own request
hook the way you like.

    from margAI import Wrapper, install_bangtags

    app = Wrapper.from_config(load_config())
    install_bangtags(app)           # layer-in !margAI: parsing

    @app.before("refine")
    def refine(ctx, tag):
        ...                         # customize here
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .core.events import STATE_KEY

__all__ = [
    "BANGTAG_NS",
    "STATE_KEY",
    "Tag",
    "find_directive",
    "install_bangtags",
    "remove_directive",
]

BANGTAG_NS = "margAI"

_DIRECTIVE_RE = re.compile(r"(?<!\S)!([A-Za-z0-9_-]+)\s*:\s*(?P<tags>[^\n]+)")


@dataclass(frozen=True)
class Tag:
    """A single parsed bangtag."""

    name: str
    value: str | None = None

    def __str__(self) -> str:
        return f"{self.name}={self.value}" if self.value else self.name


def find_directive(text: str) -> tuple[str | None, list[Tag]]:
    """Return ``(directive_text, tags)`` or ``(None, [])`` when absent."""
    if not text:
        return None, []
    match = _DIRECTIVE_RE.search(text)
    if match is None or match.group(1).lower() != BANGTAG_NS.lower():
        return None, []
    return match.group(0), _parse_tags(match.group("tags"))


def _parse_tags(raw: str) -> list[Tag]:
    tags: list[Tag] = []
    for part in raw.split(","):
        for chunk in part.split():
            name, sep, value = chunk.partition("=")
            name = name.strip().lower()
            if not name:
                continue
            tags.append(Tag(name=name, value=value.strip() or None if sep else None))
    return tags


def remove_directive(text: str, directive: str) -> str:
    """Remove a directive from the prompt (default behavior before upstream)."""
    if not directive:
        return text
    return re.sub(rf"\s*{re.escape(directive)}\s*", "", text).strip()


def install_bangtags(
    app: object,
    *,
    namespace: str = BANGTAG_NS,
    state_key: str = STATE_KEY,
    order: int = -100,
) -> None:
    """Wire bangtag parsing into ``app``'s request phase (idempotent).

    Adds a request hook that finds ``!{namespace}: ...`` in the last user
    message, strips the directive from the prompt, and stores the parsed
    ``Tag``\\ s in ``ctx.state[state_key]`` -- the key the wrapper's event
    dispatch reads, so ``@app.before("refine")``-style handlers fire.

    The hook runs at ``order``, which defaults to before the wrapper's own
    event dispatch, so tags are ready by the time ``before`` events (and
    therefore every marglet) run.
    """
    if getattr(app, "_bangtags_installed", False):
        return
    app._bangtags_installed = True
    ns = namespace.lower()

    @app.before(name=f"bangtags:{ns}", order=order)
    def parse_bangtags(ctx: object) -> None:  # pragma: no cover - trivial
        directive, tags = find_directive(_last_user_message(ctx) or "")
        if directive:
            _strip_directive(ctx, directive)
        ctx.state[state_key] = tags
        ctx.state.setdefault("margAI", {})["directive"] = directive


def _last_user_message(ctx: object) -> str | None:
    for msg in reversed(ctx.messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _strip_directive(ctx: object, directive: str) -> None:
    for msg in reversed(ctx.messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            msg["content"] = remove_directive(msg["content"], directive)
            return