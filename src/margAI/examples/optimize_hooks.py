"""A marglet that does all four phases: ``!margAI: optimize``.

    "make this faster  !margAI: optimize"

The tag is stripped from the prompt before it reaches the provider, the system
prompt is prepended, the response is tidied, the stream is annotated, and
failures come back as a clean, typed error.

Note the syntax: ``!margAI: optimize``, not ``!optimize``. The namespace is
what keeps user-typed text from accidentally firing someone's handler.
"""

from __future__ import annotations

from typing import Any

from margAI import Marglet, Wrapper, install_bangtags

OPTIMIZE_SYSTEM_PROMPT = """You are an expert code optimizer. When the user adds !margAI: optimize
to their prompt, they want you to focus on:
- Performance improvements (time/space complexity)
- Memory efficiency
- Clean, maintainable code
- Modern best practices

Provide the optimized code with brief explanations of what changed and why."""


def optimize_before(ctx: Any, tag: Any) -> None:
    """Prepend the optimizer's instructions, and keep the original prompt.

    The original is stashed because the directive is stripped from the message
    before it goes upstream, so `ctx.last_user_message()` afterwards is not
    what the user actually typed.
    """
    original = ctx.last_user_message()
    if original:
        ctx.state["optimize_original"] = original
    ctx.add_system_prompt(OPTIMIZE_SYSTEM_PROMPT, prepend=True)


def optimize_after(payload: dict, ctx: Any, tag: Any) -> dict:
    """Non-streaming: trim stray whitespace and any leaked internal markers."""
    message = payload.get("choices", [{}])[0].get("message")
    if message and isinstance(message.get("content"), str):
        message["content"] = message["content"].replace("[[optimize]]", "").strip()
    return payload


def optimize_stream(chunk: dict, ctx: Any, tag: Any, scratch: dict) -> dict | None:
    """Per-chunk: count chunks for telemetry, and drop empty ones.

    Returning ``None`` drops the chunk, which is how you keep SSE noise
    (empty keepalive deltas) out of the client's stream.
    """
    scratch["chunks"] = scratch.get("chunks", 0) + 1
    choices = chunk.get("choices") or []
    if choices and not choices[0].get("delta", {}).get("content") and not choices[0].get("finish_reason"):
        return None  # an empty delta carries nothing; drop it
    return chunk


def optimize_error(exc: Exception, ctx: Any, tag: Any) -> dict | None:
    """Failures come back typed and named, instead of leaking a traceback.

    Returning a dict takes over the error body; returning ``None`` leaves the
    default one in place.
    """
    return {
        "error": {
            "message": f"Optimization request failed: {exc}",
            "type": "optimization_error",
            "param": None,
            "code": None,
        }
    }


def register(app: Wrapper) -> Wrapper:
    """Register the optimize marglet. Idempotent, so config reloads are safe."""
    if getattr(app, "_optimize_registered", False):
        return app

    install_bangtags(app)
    app.add_marglet(
        Marglet(
            "optimize",
            summary="Refactor for performance and explain the changes",
            before=optimize_before,
            after=optimize_after,
            stream=optimize_stream,
            error=optimize_error,
        )
    )
    app._optimize_registered = True
    return app
