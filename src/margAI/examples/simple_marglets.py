"""Marglets, the short version.

Three ways to register the same marglet, all equivalent. Load with:

    [hooks]
    load = ["examples.simple_marglets"]

or programmatically:

    from examples.simple_marglets import register
    register(wrapper)

Then ask for it per call, inline, with a bangtag::

    "make it a checklist  !margAI: bullets"
"""

from __future__ import annotations

from typing import Any

from margAI import Marglet, RequestContext, Tag, Wrapper, install_bangtags

BULLETS_SYSTEM_PROMPT = (
    "Structure your answer as a bulleted list. One point per bullet, no prose paragraphs."
)


# -- 1. decorator ----------------------------------------------------------


def register_decorator(app: Wrapper) -> None:
    """The shortest form: a name, and a before-hook."""

    @app.marglet("bullets", summary="Answer as a bulleted list")
    def bullets(ctx: RequestContext, tag: Tag) -> None:
        ctx.add_system_prompt(BULLETS_SYSTEM_PROMPT)


# -- 2. explicit phases ---------------------------------------------------


def bullets_after(payload: dict, ctx: Any, tag: Any) -> dict:
    """Non-streaming: make sure it really is a list before returning it."""
    content = payload.get("choices", [{}])[0].get("message", {}).get("content")
    if content and not content.lstrip().startswith(("*", "-", "1.")):
        payload["choices"][0]["message"]["content"] = "\n".join(
            f"* {line}" for line in content.splitlines() if line.strip()
        )
    return payload


def register_explicit(app: Wrapper) -> None:
    """Every phase named explicitly, when the marglet does more than one thing."""
    app.add_marglet(
        Marglet(
            "bullets",
            summary="Answer as a bulleted list",
            before=lambda ctx, tag: ctx.add_system_prompt(BULLETS_SYSTEM_PROMPT),
            after=bullets_after,
        )
    )


# -- 3. a group of related marglets ---------------------------------------


class Writing:
    """Several marglets in one object, named ``{marglet}_{phase}``."""

    def terse_before(self, ctx: RequestContext, tag: Tag) -> None:
        """Answer in as few words as possible."""
        ctx.add_system_prompt("Answer in as few words as possible.")

    def terse_after(self, payload: dict[str, Any], ctx: RequestContext, tag: Tag) -> dict[str, Any]:
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if content:
            payload["choices"][0]["message"]["content"] = content.strip()
        return payload

    def cite_before(self, ctx: RequestContext, tag: Tag) -> None:
        """Cite a source for every claim."""
        ctx.add_system_prompt("Cite a source for every claim you make.")


def register(app: Wrapper) -> Wrapper:
    """The example's entry point: ``[hooks] load = [...]`` calls this."""
    install_bangtags(app)  # make "!margAI: ..." work at all
    register_decorator(app)
    app.add_marglet_group(Writing())
    return app
