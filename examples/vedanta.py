"""Vedanta example: application bangtags layered on top of margAI.

Two ways to use this file:

- **As a hook module.** It exposes ``register(app)``, the single entry point
  margAI's ``[hooks] load`` looks for. Point ``examples/vedanta.toml`` at it
  with ``load = ["vedanta"]`` and run the gateway from the ``examples``
  directory (so ``vedanta`` is importable), or copy the module into your own
  package and use its dotted path.
- **As a runnable script.** ``uv run python examples/vedanta.py`` builds the
  wrapper from ``examples/vedanta.toml``, installs these handlers, and serves
  on the host/port that file declares.

The handlers are namespaced to the gateway prefix (``veda``), so a client
opts in with ``!veda: refine structure``. margAI's own routing tags stay under
``!margAI:`` and the two namespaces never see each other's directives -- which
is what keeps a stray word in a user's prompt from firing a handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from weakref import WeakSet

from margAI import Wrapper, install_bangtags

__all__ = ["annotate", "bulletize", "compact", "refine", "register", "structure"]


def compact(text: str) -> str:
    """Collapse whitespace runs.

    Stands in for whatever prompt compaction your application actually does;
    margAI supplies the pipeline, the text work is yours.
    """
    return " ".join(text.split())


def bulletize(text: str) -> str:
    """Turn each non-empty line into a bullet."""
    return "\n".join(f"- {line.strip()}" for line in text.splitlines() if line.strip())


def refine(ctx: Any, tag: Any) -> None:
    """``!veda: refine`` -- stash a compacted copy of the user's message.

    The directive is stripped from the prompt before it reaches the provider,
    so ``ctx.last_user_message()`` is still the user's own text here; the copy
    is kept because later handlers may want the pre-directive form.
    """
    ctx.state["refined"] = compact(ctx.last_user_message() or "")


def structure(payload: dict[str, Any], ctx: Any, tag: Any) -> dict[str, Any]:
    """``!veda: structure`` -- bullet the first choice's content.

    Defensive about the shape: an error path or an upstream that returns no
    choices must pass through untouched rather than turn a formatting nicety
    into a 500.
    """
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError):
        return payload
    if isinstance(message.get("content"), str):
        message["content"] = bulletize(message["content"])
    return payload


def annotate(payload: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """Every call: mark that this example saw it.

    The per-call form (no tag), as opposed to the tag-keyed ``refine`` and
    ``structure`` above.
    """
    ctx.state["vedanta_seen"] = True
    return payload


# Weak so a discarded wrapper does not keep the module's registration alive.
_registered: WeakSet[Wrapper] = WeakSet()


def register(app: Wrapper) -> Wrapper:
    """Install the vedanta handlers. Idempotent, so config reloads are safe."""
    if app in _registered:
        return app

    # One namespace per pack: claim "!veda: ..." while margAI keeps "!margAI:".
    install_bangtags(app, namespace=app.namespace)
    app.before("refine")(refine)
    app.after("structure")(structure)
    app.after(annotate)

    _registered.add(app)
    return app


def main() -> None:
    """Build, register, and serve -- the ``python vedanta.py`` entry point."""
    import uvicorn

    from margAI.config import load_config
    from margAI.transport.fastapi import build_app

    config = load_config(Path(__file__).resolve().parent / "vedanta.toml")
    wrapper = Wrapper.from_config(config)
    register(wrapper)
    uvicorn.run(build_app(wrapper=wrapper), host=config.gateway.host, port=config.gateway.port)


if __name__ == "__main__":
    main()
