"""The dynamic router: choosing a provider+model per call.

``margAI/dynamic`` is a reserved model id. When a client asks for it, the
call is not routed by name -- it is routed by *policy*, chosen at request
time from an ordered chain of selectors::

    intent = RoutingIntent.from_tags(tags)     # "!margAI: cheap, not=openai"
    intent = marglet before-hooks refine it    # ctx.intent.max_cost_per_1m = 0.5
    candidates = router.candidates()           # every configured (provider, model)
    candidates = intent.filter(candidates)     # declarative constraints
    for selector in chain:                     # first non-None wins
        route = selector(intent, candidates, ctx)
        if route is not None:
            break

A selector is any callable ``fn(intent, candidates, ctx)`` returning a
:class:`~margAI.core.router.Route`, a :class:`~margAI.core.router.Candidate`,
a ``"provider/model"`` string, ``None`` to abstain, or a list of any of those
(meaning "any of these will do" -- the chain takes the first and moves on).
Returning ``None`` is the normal case and never an error.

Because abstention is normal, *no selector* is not a failure either: the chain
falls back to ``default_provider`` and says so in the route's ``reason``. What
*is* an error is having nothing to fall back to, and that produces a 404 that
names the problem instead of "Unknown model".

Determinism is a feature. Given the same config, the same tags, and the same
marglets, the same route comes out -- so a ``dynamic`` call is debuggable
rather than a coin flip. If you want load spreading across candidates, do it
inside a selector, explicitly, and put the policy in the ``reason``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .core.errors import ApiError
from .core.intent import RoutingIntent
from .core.router import Candidate, ModelRouter, Route

__all__ = ["Routing", "Selector", "coerce_route"]

logger = logging.getLogger("margAI.routing")

SelectorFn = Callable[[RoutingIntent, Sequence[Candidate], Any], Any]


@dataclass
class Selector:
    """One named policy in the chain."""

    name: str
    fn: SelectorFn
    order: int = 0
    seq: int = 0
    only_for: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)

    def __call__(self, intent: RoutingIntent, candidates: Sequence[Candidate], ctx: Any) -> Any:
        return self.fn(intent, candidates, ctx)

    def applies_to(self, intent: RoutingIntent) -> bool:
        """A selector scoped with ``only_for`` only sees the calls carrying
        one of those tags -- the mechanism behind "marglet-scoped routing"."""
        if not self.only_for:
            return True
        return bool(self.only_for.intersection(intent.tags))


def coerce_route(value: Any, *, selector: str = "") -> Route | None:
    """Normalise whatever a selector returned into a :class:`Route`.

    Returns ``None`` for an abstention. Raises for something that is clearly a
    mistake (a bad string, a dict with no provider), because a silently
    ignored selector is worse than a loud one.
    """
    if value is None:
        return None
    if isinstance(value, Route):
        return value
    if isinstance(value, Candidate):
        return Route(value.provider, value.model, f"selector:{selector}" if selector else "")
    if isinstance(value, str):
        provider, sep, model = value.partition("/")
        if not sep or not provider or not model:
            raise ApiError(
                500,
                f"selector {selector!r} returned {value!r}; expected "
                f"'<provider>/<model>', a Candidate, or a Route",
                error_type="server_error",
            )
        return Route(provider, model, f"selector:{selector}" if selector else "")
    if isinstance(value, dict):
        provider = value.get("provider")
        model = value.get("model")
        if not provider or not model:
            raise ApiError(
                500,
                f"selector {selector!r} returned a dict without provider/model: {value!r}",
                error_type="server_error",
            )
        return Route(str(provider), str(model), f"selector:{selector}" if selector else "")
    if isinstance(value, (list, tuple)):
        for item in value:
            route = coerce_route(item, selector=selector)
            if route is not None:
                return route
        return None
    raise ApiError(
        500,
        f"selector {selector!r} returned an unsupported type: {type(value).__name__}",
        error_type="server_error",
    )


class Routing:
    """The selector chain, and the entry point the wrapper calls.

    Lives on the wrapper as ``wrapper.routing``.
    """

    def __init__(self, router: ModelRouter, *, default_provider: str | None = None) -> None:
        self.router = router
        self.default_provider = default_provider
        self._selectors: list[Selector] = []

    # -- registration -------------------------------------------------------

    def add_selector(
        self,
        fn: SelectorFn,
        *,
        name: str | None = None,
        order: int = 0,
        only_for: Iterable[str] = (),
    ) -> Selector:
        """Append a policy to the chain.

        ``order`` decides precedence (lower runs first, and ties break by
        registration). ``only_for`` scopes the policy to calls carrying those
        tags, which is how a marglet gets to own routing for the calls it is
        active on.
        """
        selector = Selector(
            name=name or getattr(fn, "__name__", None) or f"selector{len(self._selectors)}",
            fn=fn,
            order=order,
            seq=len(self._selectors),
            only_for=frozenset(only_for),
        )
        self._selectors.append(selector)
        return selector

    def selector(self, fn: SelectorFn | None = None, *, name: str | None = None, order: int = 0, only_for: Iterable[str] = ()):
        """Decorator form of :meth:`add_selector`."""
        if fn is None:
            def deco(inner: SelectorFn) -> SelectorFn:
                self.add_selector(inner, name=name, order=order, only_for=only_for)
                return inner

            return deco
        return self.add_selector(fn, name=name, order=order, only_for=only_for)

    def remove_selector(self, name: str) -> Selector | None:
        """Drop a selector by name, returning it so the caller can keep a
        reference to it. Removing an unknown name is a no-op, not an error --
        it makes ``replace=True``-style code paths simple."""
        removed = next((s for s in self._selectors if s.name == name), None)
        if removed is not None:
            self._selectors = [s for s in self._selectors if s.name != name]
        return removed

    def selectors(self) -> list[Selector]:
        """Chain order: ``order`` first, then registration sequence."""
        return sorted(self._selectors, key=lambda s: (s.order, s.seq))

    def names(self) -> list[str]:
        return [s.name for s in self.selectors()]

    def __len__(self) -> int:
        return len(self._selectors)

    # -- resolution ---------------------------------------------------------

    def candidates(self, intent: RoutingIntent | None = None) -> list[Candidate]:
        """Every configured (provider, model) pair, filtered by the intent."""
        cands = self.router.candidates()
        return intent.filter(cands) if intent is not None else cands

    def select(self, intent: RoutingIntent, ctx: Any = None) -> Route:
        """Resolve a dynamic request. Raises :class:`ApiError` if nothing can
        serve it, with a message that says what was tried."""
        candidates = self.candidates(intent)
        if not candidates:
            raise ApiError(
                404,
                self._empty_message(intent, "no configured (provider, model) pair matches this call's constraints"),
                error_type="invalid_request_error",
                param="model",
            )

        considered: list[str] = []
        for selector in self.selectors():
            if not selector.applies_to(intent):
                continue
            considered.append(selector.name)
            raw = selector(intent, candidates, ctx)
            # A list means "any of these will do", so it is a preference
            # order: walk it until one entry is actually routable. Treating
            # only the first entry as the answer would throw away the rest
            # whenever the first happened to be stale.
            for value in raw if isinstance(raw, (list, tuple)) else [raw]:
                route = coerce_route(value, selector=selector.name)
                if route is None:
                    continue
                route = self._validate(route, intent, candidates, selector.name)
                if route is not None:
                    logger.debug(
                        "dynamic route: %s/%s via %s (intent=%s)",
                        route.provider,
                        route.model,
                        route.reason,
                        intent.describe(),
                    )
                    return route

        return self._fallback(intent, candidates, considered)

    def _validate(
        self, route: Route, intent: RoutingIntent, candidates: Sequence[Candidate], selector: str
    ) -> Route | None:
        """A selector may return something that isn't actually routable (a
        typo'd provider, a model nobody configured). Reject it and keep going
        rather than failing the call -- a bad policy shouldn't take down
        traffic, but it should be visible."""
        known = {(c.provider, c.model) for c in candidates}
        if (route.provider, route.model) in known:
            return route
        # A default model per provider is legitimate even when not enumerated.
        provider = self.router.providers.get(route.provider)
        if provider is not None and route.model == getattr(provider.config, "default_model", None):
            return route
        logger.warning(
            "selector %r chose %s/%s, which is not a configured candidate; ignoring",
            selector,
            route.provider,
            route.model,
        )
        return None

    def _fallback(self, intent: RoutingIntent, candidates: Sequence[Candidate], considered: list[str]) -> Route:
        if self.default_provider:
            provider = self.router.providers.get(self.default_provider)
            model = getattr(getattr(provider, "config", None), "default_model", None)
            if model:
                # A fallback that violates the call's own constraints is not a
                # fallback, it is a bug report: "!margAI: not=openai" must not
                # quietly be served by openai. Checked against the constraints
                # directly, not against candidate enumeration -- a default
                # model is legitimately absent from the configured list.
                if not intent.allows(self.default_provider, model):
                    raise ApiError(
                        404,
                        self._empty_message(
                            intent,
                            f"default_provider is {self.default_provider!r} but this call's constraints "
                            f"exclude it",
                        ),
                        error_type="invalid_request_error",
                        param="model",
                    )
                if considered:
                    logger.info(
                        "no dynamic selector claimed this call (tried: %s); using default_provider",
                        ", ".join(considered),
                    )
                return Route(self.default_provider, model, "default_provider")
        raise ApiError(
            404,
            self._empty_message(
                intent,
                f"no dynamic selector produced a route (tried: {', '.join(considered) or 'none registered'}) "
                f"and no default_provider is configured",
            ),
            error_type="invalid_request_error",
            param="model",
        )

    def _empty_message(self, intent: RoutingIntent, detail: str) -> str:
        return (
            f"Cannot route {self.router.dynamic_id!r}: {detail}. "
            f"Intent: {intent.describe()}. Candidates: "
            f"{[f'{c.provider}/{c.model}' for c in self.router.candidates()][:12] or 'none'}. "
            f"Register a selector with wrapper.routing.add_selector(...) or set gateway.default_provider."
        )

    # -- introspection ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "dynamic_id": self.router.dynamic_id,
            "selectors": [
                {"name": s.name, "order": s.order, "only_for": sorted(s.only_for)} for s in self.selectors()
            ],
            "candidates": [f"{c.provider}/{c.model}" for c in self.router.candidates()],
            "default_provider": self.default_provider,
        }
