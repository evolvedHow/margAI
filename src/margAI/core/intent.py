"""Routing intent: what a call wants from the router, gathered before resolution.

Pure stdlib. The intent is seeded from the call's bangtags and then refined by
marglet ``before`` hooks, so by the time the dynamic router runs, everything
that wanted a say has said it::

    # via a bangtag
    "draw it  !margAI: dynamic, cheap"
    # -> intent.tags == ("dynamic", "cheap")

    # via a marglet hook (immutable, so refine and reassign)
    ctx.steer(provider="local", max_cost_per_1m=0.5)

Selectors (:mod:`margAI.routing`) read the intent and return a route. They
never mutate it -- an intent that changes under a selector is how "why did it
pick that?" becomes unanswerable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["INTENT_KEY", "RoutingIntent"]

# Where the intent lives on the request context's state dict.
INTENT_KEY = "marglets.intent"


@dataclass(frozen=True)
class RoutingIntent:
    """An immutable snapshot of a call's routing preferences.

    Marglets refine it with :meth:`with_` (``ctx.intent = ctx.intent.with_(...)``)
    rather than mutating, so an intent handed to a selector is a stable value
    that can be logged and compared after the fact.
    """

    model: str | None = None
    provider: str | None = None
    tags: tuple[str, ...] = ()
    exclude: frozenset[str] = frozenset()
    require: frozenset[str] = frozenset()
    max_cost_per_1m: float | None = None
    notes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_tags(cls, tags: Iterable[Any]) -> RoutingIntent:
        """Seed an intent from parsed bangtags.

        Recognised tag values become first-class fields so selectors do not
        have to re-parse strings:

        - ``route=<provider/model>`` or ``route=<model>`` -> ``provider``/``model``
        - ``provider=<name>`` -> ``provider``
        - ``model=<name>`` -> ``model``
        - ``not=<provider/model>`` -> ``exclude``
        - ``only=<provider/model>`` -> ``require``

        Unrecognised values are left alone: they stay in ``tags`` and are the
        selector's business, which is the whole point of the tag namespace.
        """
        provider = model = None
        exclude: set[str] = set()
        require: set[str] = set()
        names: list[str] = []
        for tag in tags:
            name = getattr(tag, "name", None)
            if not name:
                continue
            names.append(name)
            value = getattr(tag, "value", None)
            if not value:
                continue
            if name == "route":
                if "/" in value:
                    provider, _, model = value.partition("/")
                else:
                    model = value
            elif name == "provider":
                provider = value
            elif name == "model":
                model = value
            elif name == "not":
                exclude.add(value)
            elif name == "only":
                require.add(value)
        return cls(
            model=model,
            provider=provider,
            tags=tuple(names),
            exclude=frozenset(exclude),
            require=frozenset(require),
        )

    def with_(self, **changes: Any) -> RoutingIntent:
        return replace(self, **changes)

    def wants(self, tag: str) -> bool:
        return tag in self.tags

    def note(self, key: str, value: Any) -> RoutingIntent:
        """Attach an advisory note (merged with any existing value for ``key``)."""
        merged = {k: v for k, v in self.notes if k != key}
        merged[key] = value
        return self.with_(notes=tuple(sorted(merged.items(), key=lambda kv: kv[0])))

    def allows(self, provider: str, model: str) -> bool:
        """Whether this intent's *declarative constraints* permit a pair.

        A constraint token (``not=``/``only=``) matches a candidate when it
        names the provider, the model, or the ``provider/model`` pair. All
        three are accepted because ``not=openai`` is what people actually
        write, and matching only the pair or the model would let it through
        silently.

        This is the constraint question only -- it says nothing about whether
        the pair is configured or enumerated. Kept separate on purpose:
        :meth:`filter` needs both, but the ``default_provider`` fallback needs
        only this, because serving a provider's default model is legitimate
        even when no configured model list mentions it. Folding the two
        together let a default model slip past an explicit ``not=``.
        """
        if self._excluded(provider, model):
            return False
        if self.provider and provider != self.provider:
            return False
        if self.model and model != self.model:
            return False
        if self.require and not self._required(provider, model):
            return False
        return True

    @staticmethod
    def _named(tokens: frozenset[str], provider: str, model: str) -> bool:
        """Whether any constraint token names this candidate's provider, its
        model, or the pair."""
        return provider in tokens or model in tokens or f"{provider}/{model}" in tokens

    def _excluded(self, provider: str, model: str) -> bool:
        return self._named(self.exclude, provider, model)

    def _required(self, provider: str, model: str) -> bool:
        return self._named(self.require, provider, model)

    def filter(self, candidates: Iterable[Any]) -> list[Any]:
        """Apply the declarative constraints. The predictable part; selectors
        do the interesting work."""
        return [
            cand
            for cand in candidates
            if self.allows(getattr(cand, "provider", ""), getattr(cand, "model", ""))
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "tags": list(self.tags),
            "exclude": sorted(self.exclude),
            "require": sorted(self.require),
            "max_cost_per_1m": self.max_cost_per_1m,
            "notes": dict(self.notes),
        }
