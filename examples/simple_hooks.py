"""Trivial demonstration hooks (no bangtags -- that is app #1, built later).

Wires one request hook, one response hook, and one stream hook onto the
wrapper. Load with:

    [hooks]
    load = ["examples.simple_hooks"]

or programmatically:

    from examples.simple_hooks import register
    register(wrapper)
"""

from margai import Wrapper


def register(wrapper: Wrapper) -> Wrapper:
    @wrapper.request
    def note_english(ctx):
        """Annotate the last user message with a stable marker (state)."""
        ctx.state["seen_marker"] = bool("marker" in (ctx.last_user_message() or ""))

    @wrapper.response
    def stamp_label(payload, ctx):
        """Append a footer to non-streamed completions."""
        if payload.get("choices"):
            content = payload["choices"][0].get("message", {}).get("content", "")
            payload["choices"][0]["message"]["content"] = content + "\n\n[framework demo hook]"
        return payload

    @wrapper.stream
    def stamp_stream(chunk, ctx):
        """Append the footer to the final streaming chunk."""
        choices = chunk.get("choices") or []
        if choices and choices[0].get("finish_reason") == "stop":
            content = choices[0].get("delta", {}).get("content", "")
            if content:
                choices[0]["delta"]["content"] = content + "\n\n[framework demo hook]"
        return chunk

    return wrapper