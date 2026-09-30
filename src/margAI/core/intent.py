"""Routing intent: what a call wants from the router, gathered before resolution.

Pure stdlib. The intent is seeded from the call's bangtags and then refined by
``before`` handlers, so by the time the dynamic router runs, everything that
wanted a say has said it::

    # via a bangtag
    "draw it  !margAI: dynamic, cheap"
    # -> intent.tags == ("dynamic", "cheap")

    # via a handler (immutable, so refine and reassign)
    ctx.steer(provider="local", max_cost_per_1m=0.5)

Selectors (:mod:`margAI.routing`) read the intent and return a route. They
never mutate it -- an intent that changes under a selector is how "why did it
pick that?" becomes unanswerable.

**This module knows no tag names.** :meth:`RoutingIntent.from_tags` records
every tag into an open vocabulary and stops there. What ``route=`` or
``not=`` *mean* is declared by handlers in :mod:`margAI.builtin_tags`, like any
other pack, which is why adding a directive needs no change here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .tags import qualify

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
    qualified: tuple[str, ...] = ()
    selectors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    exclude: frozenset[str] = frozenset()
    require: frozenset[str] = frozenset()
    max_cost_per_1m: float | None = None
    notes: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        # Selectors are read by selectors and written by handlers, and the
        # dataclass is frozen so the mapping is the only thing that could drift.
        object.__setattr__(self, "selectors", MappingProxyType(dict(self.selectors)))

    @classmethod
    def from_tags(cls, tags: Iterable[Any]) -> RoutingIntent:
        """Seed an intent from parsed bangtags, naming nothing.

        Every tag lands in :attr:`tags`, every valued tag in
        :attr:`selectors`, and every namespaced tag in :attr:`qualified`. The
        typed fields stay empty on purpose: something has to *decide* that
        ``route=`` means a model, and that something is a handler in
        :mod:`margAI.builtin_tags` -- the same place a third-party pack puts
        its own directives. Seeding it here instead is what made the tag
        vocabulary a closed set.
        """
        names: list[str] = []
        qualified: list[str] = []
        selectors: dict[str, list[str]] = {}
        for tag in tags:
            name = getattr(tag, "name", None)
            if not name:
                continue
            names.append(name)
            full = qualify(tag)
            if full and ":" in full:
                qualified.append(full)
            value = getattr(tag, "value", None)
            if value:
                selectors.setdefault(name, []).append(value)
        return cls(
            tags=tuple(names),
            qualified=tuple(qualified),
            selectors={name: tuple(values) for name, values in selectors.items()},
        )

    def with_(self, **changes: Any) -> RoutingIntent:
        return replace(self, **changes)

    def wants(self, tag: str) -> bool:
        return tag in self.tags

    def values_for(self, name: str) -> tuple[str, ...]:
        """Every value given to selector ``name``, in the order written.

        Empty when the tag appeared bare (``!margAI: cheap``) or not at all,
        which is why a handler that needs a value checks :meth:`has` first or
        uses :meth:`value`.
        """
        return self.selectors.get(name, ())

    def has(self, name: str) -> bool:
        """Whether selector ``name`` appeared at all, valued or bare."""
        return name in self.tags

    def value(self, name: str, default: str | None = None) -> str | None:
        """The first value given to selector ``name``."""
        for candidate in self.values_for(name):
            return candidate
        return default

    def from_namespace(self, namespace: str) -> tuple[str, ...]:
        """Qualified names belonging to ``namespace`` -- what a pack's
        selectors filter on without string surgery."""
        from .tags import namespace_key

        prefix = f"{namespace_key(namespace)}:"
        return tuple(name for name in self.qualified if name.startswith(prefix))

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
            "qualified": list(self.qualified),
            "selectors": {name: list(values) for name, values in self.selectors.items()},
            "exclude": sorted(self.exclude),
            "require": sorted(self.require),
            "max_cost_per_1m": self.max_cost_per_1m,
            "notes": dict(self.notes),
        }
