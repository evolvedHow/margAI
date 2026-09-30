"""The tags margAI ships with -- registered as a pack, not as special cases.

Every bangtag the framework understands lives here, and nothing else in the
codebase knows any of their names. ``!margAI: route=local/qwen3`` works because
of a handler in this file, declared with the same :meth:`~margAI.Wrapper.on`
decorator a third-party pack uses, installed through the same
:meth:`~margAI.Wrapper.pack` context manager, parsed by the same
:func:`~margAI.bangtag.install_bangtags` hook. If this file were deleted, the
framework would still run -- ``!margAI: route=`` would simply become an unknown
tag, and so would every documented example.

That is the point. The alternative, teaching the router that ``route`` means a
model, is what makes a tag vocabulary a closed set: every new directive would
need a change in the orchestration layer, and a pack could not add one.

The two axes are kept apart on purpose:

- **Selectors** (``route``, ``provider``, ``model``, ``not``, ``only``,
  ``cost``) refine :class:`~margAI.core.intent.RoutingIntent`, which the
  dynamic router reads when a call asks for ``margAI/dynamic``.
- **Parameters** (``think``) write to the request body, and mean nothing to
  routing at all.

The ``order`` on each handler is the precedence between them, lower first. A
bare ``!margAI: think`` sets a middle reasoning effort; ``think=high`` overrides
it. Likewise ``model=`` is applied after ``route=`` so the more specific
selector wins, and ``not=``/``only=`` are applied first so a pin never has to
reason about whether it is about to be contradicted.
"""

from __future__ import annotations

from typing import Any

from .bangtag import install_bangtags
from .core.errors import ApiError
from .core.tags import RESERVED_NAMESPACE

__all__ = ["CORE_PACK", "install"]

#: Pack id. Recognised by the registry as framework-owned, which is what lets
#: it register into the reserved namespace where a third-party pack may not.
CORE_PACK = "margAI.builtin_tags"

#: Reasoning efforts accepted by `think=`, worst to best.
_EFFORTS = ("low", "medium", "high")

# Selector precedence, applied in this order. Policy before pin, coarse before
# fine, so a call that says both is answered by the narrower claim.
_ORDER_POLICY = -30
_ORDER_ROUTE = -28
_ORDER_PROVIDER = -27
_ORDER_MODEL = -26
_ORDER_COST = -25
_ORDER_PARAM = -20


def _bad_value(name: str, wanted: str, got: str | None) -> ApiError:
    """A bad selector value is the caller's mistake, so it is a 400.

    Raised as an :class:`ApiError` rather than a bare ``ValueError``: a
    handler raising the latter is reported as an unhandled 500, which turns
    ``!margAI: think=extreme`` into "Internal server error" and tells the
    caller nothing about the tag they got wrong.
    """
    return ApiError(
        400,
        f"!margAI: {name}= wants {wanted}; got {got!r}",
        error_type="invalid_request_error",
        param="bangtags",
    )


def install(app: Any) -> None:
    """Install the framework's own pack into ``app``.

    Idempotent: parsing is guarded per app and namespace by
    :func:`~margAI.bangtag.install_bangtags`, and every handler below is
    registered as the application already owning ``margAI``'s namespace -- the
    one pack allowed there -- so a second call replaces its own handlers rather
    than stacking a second copy of each.

    Safe to call from application code, which is also how you *can* displace a
    built-in: register your own handler for the same tag from your own code
    and pass ``replace=True``.
    """
    install_bangtags(app, namespace=RESERVED_NAMESPACE)

    with app.pack(CORE_PACK, namespace=RESERVED_NAMESPACE, doc="Routing selectors and request parameters."):

        # -- policy: applied before any pin -------------------------------

        @app.on("not", order=_ORDER_POLICY)
        def exclude(ctx: Any, tag: Any) -> None:
            """`not=<provider|model|provider/model>` -- forbid a target.

            Repeatable. Matches a provider, a model, or the pair, because
            `not=openai` is what people actually write.
            """
            ctx.intent = ctx.intent.with_(exclude=ctx.intent.exclude | {tag.value})

        @app.on("only", order=_ORDER_POLICY + 1)
        def require(ctx: Any, tag: Any) -> None:
            """`only=<provider|model|provider/model>` -- restrict to a target.

            Repeatable. A call constrained to several targets is allowed to
            use any one of them.
            """
            ctx.intent = ctx.intent.with_(require=ctx.intent.require | {tag.value})

        # -- pins ---------------------------------------------------------

        @app.on("route", order=_ORDER_ROUTE)
        def route(ctx: Any, tag: Any) -> None:
            """`route=<model>` or `route=<provider>/<model>` -- pin a target.

            This is a *preference*, not a bypass: the pin is still checked
            against the call's own `not=`/`only=` policy and fails loudly if it
            contradicts it, rather than quietly serving the excluded model.
            A bare `route=` with no provider pins the model across providers.
            """
            provider, sep, model = tag.value.partition("/")
            if sep:
                ctx.intent = ctx.intent.with_(provider=provider, model=model or None)
            else:
                ctx.intent = ctx.intent.with_(model=provider)

        @app.on("provider", order=_ORDER_PROVIDER)
        def provider(ctx: Any, tag: Any) -> None:
            """`provider=<name>` -- pin the provider, leaving the model to routing."""
            ctx.intent = ctx.intent.with_(provider=tag.value)

        @app.on("model", order=_ORDER_MODEL)
        def model(ctx: Any, tag: Any) -> None:
            """`model=<name>` -- pin the model. Applied after `route=`, so the
            narrower of the two wins."""
            ctx.intent = ctx.intent.with_(model=tag.value)

        @app.on("cost", order=_ORDER_COST)
        def cost(ctx: Any, tag: Any) -> None:
            """`cost=<usd per 1m tokens>` -- the ceiling a selector may price against."""
            try:
                limit = float(tag.value)
            except (TypeError, ValueError):
                raise _bad_value("cost", "a number", tag.value) from None
            ctx.intent = ctx.intent.with_(max_cost_per_1m=limit)

        # -- request parameters -------------------------------------------

        @app.on("think", order=_ORDER_PARAM)
        def think(ctx: Any, tag: Any) -> None:
            """`think` or `think=<low|medium|high>` -- reasoning effort.

            A routing-neutral parameter: it shapes the request body and is
            invisible to the selector chain, which is the whole reason the
            two axes are separate.
            """
            effort = tag.value or "medium"
            if effort not in _EFFORTS:
                raise _bad_value("think", f"one of {', '.join(_EFFORTS)}", tag.value)
            ctx.body["reasoning_effort"] = effort
