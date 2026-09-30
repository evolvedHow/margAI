"""Marglets: small, named enhancements that ride along with a call.

Pure stdlib (the marglet *contract*); :mod:`margAI.wiring` is what turns a
registered marglet into live hooks. A marglet is four optional phase hooks
plus metadata, grouped under one name, and the name is the whole contract:
the bangtag parser turns ``!margAI: terse`` into a tag named ``terse``, and
every marglet registered under ``terse`` fires.

So a marglet can be written three ways, all equivalent:

    # 1. decorator
    @app.marglet("terse")
    def _(ctx, tag): ctx.add_system_prompt("Answer in as few words as possible.")

    # 2. explicit phases
    app.add_marglet(Marglet("terse", before=terse, after=unpad))

    # 3. a group of related marglets
    app.add_marglet_group(Bullets())          # before_bullets, after_bullets, ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACTIVE_KEY",
    "DYNAMIC_KEY",
    "PHASES",
    "ROUTE_REASON_KEY",
    "Marglet",
    "MargletRegistry",
    "MargletSpec",
    "PhaseHooks",
]

# The four phase names. Canonical definition -- ``events.EVENTS`` and the
# wrapper's marglet wiring both alias this, so the three can't drift.
PHASES = ("before", "after", "stream", "error")

# Keys inside ``RequestContext.state`` under the ``marglets.`` namespace.
ACTIVE_KEY = "marglets.active"
DYNAMIC_KEY = "marglets.dynamic"
ROUTE_REASON_KEY = "marglets.route_reason"

PhaseHooks = dict[str, Callable[..., Any] | None]


@dataclass(frozen=True)
class MargletSpec:
    """What a front end needs to know about a marglet without running it."""

    name: str
    summary: str = ""
    order: int = 0
    tags: tuple[str, ...] = ()
    affects_routing: bool = False
    cost_estimate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "order": self.order,
            "tags": list(self.tags),
            "affects_routing": self.affects_routing,
            "cost_estimate": self.cost_estimate,
        }


class Marglet:
    """One named enhancement. Every phase is optional.

    Phases take the same arguments as the equivalent event handler:

    - ``before(ctx, tag)`` -- request phase. Mutate ``ctx.body`` / ``ctx.state``;
      refine ``ctx.intent`` to steer the dynamic router.
    - ``after(payload, ctx, tag) -> payload`` -- non-streaming response.
    - ``stream(chunk, ctx, tag, scratch) -> chunk | None`` -- per SSE chunk;
      return ``None`` to drop the chunk.
    - ``error(exc, ctx, tag) -> dict | None`` -- failure shaping; a dict takes
      over the error body.
    - ``select(intent, candidates, ctx) -> Route | None`` -- optional, and only
      consulted for the calls this marglet is active on. Makes the marglet a
      *router* for those calls. See :mod:`margAI.routing`.
    """

    def __init__(
        self,
        name: str,
        *,
        before: Callable[..., Any] | None = None,
        after: Callable[..., Any] | None = None,
        stream: Callable[..., Any] | None = None,
        error: Callable[..., Any] | None = None,
        select: Callable[..., Any] | None = None,
        summary: str = "",
        order: int = 0,
        tags: Iterable[str] = (),
        cost_estimate: float | None = None,
    ) -> None:
        if not name:
            raise ValueError("a marglet needs a name (it is the bangtag tag)")
        self.name = name
        self.order = order
        self.tags = tuple(tags)
        self.cost_estimate = cost_estimate
        self.summary = summary or _first_line(before or after or stream or error)
        self.select = select
        self.phases: PhaseHooks = {
            "before": before,
            "after": after,
            "stream": stream,
            "error": error,
        }

    @property
    def affects_routing(self) -> bool:
        return self.select is not None

    def __getitem__(self, phase: str) -> Callable[..., Any] | None:
        if phase == "select":
            return self.select
        return self.phases.get(phase)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        active = [p for p in PHASES if self.phases.get(p) is not None]
        if self.select is not None:
            active.append("select")
        return f"<Marglet {self.name!r} phases={active}>"

    def spec(self) -> MargletSpec:
        return MargletSpec(
            name=self.name,
            summary=self.summary,
            order=self.order,
            tags=self.tags,
            affects_routing=self.affects_routing,
            cost_estimate=self.cost_estimate,
        )


class MargletRegistry:
    """Name -> marglet, with the lookup rules dispatch depends on.

    Registering a second marglet under a taken name raises, rather than
    overwriting: a marglet that silently replaces another is a bug you only
    find in production. Pass ``replace=True`` when you mean it.
    """

    def __init__(self) -> None:
        self._marglets: dict[str, Marglet] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._marglets

    def __iter__(self) -> Iterator[Marglet]:
        return iter(self.ordered())

    def __len__(self) -> int:
        return len(self._marglets)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<MargletRegistry {self.names()}>"

    def get(self, name: str) -> Marglet | None:
        return self._marglets.get(name)

    def names(self) -> list[str]:
        return sorted(self._marglets)

    def ordered(self) -> list[Marglet]:
        """Dispatch order: ``order`` first, then name, so it is stable and
        does not depend on import order."""
        return sorted(self._marglets.values(), key=lambda m: (m.order, m.name))

    def add(self, marglet: Marglet, *, replace: bool = False) -> Marglet:
        existing = self._marglets.get(marglet.name)
        if existing is not None and not replace:
            raise ValueError(
                f"a marglet named {marglet.name!r} is already registered; "
                f"pass replace=True to override it"
            )
        self._marglets[marglet.name] = marglet
        return marglet

    def remove(self, name: str) -> Marglet | None:
        return self._marglets.pop(name, None)

    def active(self, tags: Iterable[Any]) -> list[Marglet]:
        """Marglets activated by ``tags``, in dispatch order, deduplicated."""
        names: list[str] = []
        for tag in tags:
            name = getattr(tag, "name", tag)
            if name in self._marglets and name not in names:
                names.append(name)
        return [self._marglets[name] for name in sorted(names, key=lambda n: (self._marglets[n].order, n))]

    def specs(self) -> list[dict[str, Any]]:
        return [m.spec().to_dict() for m in self.ordered()]

    def describe(self) -> dict[str, str]:
        return {m.name: m.summary for m in self.ordered()}


def _first_line(fn: Callable[..., Any] | None) -> str:
    if fn is None:
        return ""
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0] if doc else ""
