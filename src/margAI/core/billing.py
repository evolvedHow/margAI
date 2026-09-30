"""In-process spend tracking, for the ``highest_balance``/``least_used`` strategies.

The two strategies that depend on history need somewhere to keep it. This is
that place: an in-memory ledger of estimated spend and call counts per model,
scoped to the configured billing window.

Three things about it are worth being explicit about, because they are
assumptions the caller may not share:

- **It is an estimate.** Cost comes from the same ``[telemetry.costs]`` table
  the cost report uses, and a model with no price contributes *nothing* -- not
  a guess, not a price. So a model absent from that table looks identical to one
  that was never called, and on a gateway with no price table the whole ledger
  stays at zero. This is why ``highest_balance`` is a way of avoiding a budget
  you declared, not a way of reading a provider's account.
- **It is per-process.** Entries live in this object and nowhere else, so under
  multiple uvicorn workers each process counts only its own traffic and every
  process sees the same (partial) totals. The same caveat already applies to
  ``round_robin``'s cursor. Sharing it needs a store, not a config.
- **It is cumulative within a window.** Spend is added, never corrected, and it
  resets by the window rolling past -- not by a timer, so a gateway idle across
  a boundary simply starts the new window empty, which is the correct reading
  of a budget that has rolled over.

Nothing here does I/O and nothing is persisted, which keeps the core portable
to a Worker.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Any

__all__ = ["BudgetTable", "SpendLedger"]


class BudgetTable:
    """Declared ``$`` caps, keyed ``provider/model`` or bare ``model``.

    Mirrors :class:`~margAI.core.virtual.CostTable`: the most specific match
    wins, and an entry keyed only by model name applies to any provider
    offering it. A provider-only key covers everything that provider hosts,
    which is the shape a per-provider budget takes.
    """

    def __init__(self, entries: Iterable[Any] = ()) -> None:
        self._by_pair: dict[tuple[str, str], float] = {}
        self._by_model: dict[str, float] = {}
        self._by_provider: dict[str, float] = {}
        for entry in entries:
            model = getattr(entry, "model", None)
            provider = getattr(entry, "provider", None)
            amount = getattr(entry, "input_price", None)
            if amount is None:
                continue
            if model in (None, "*") and provider not in (None, "*", ""):
                # provider/<anything> -- a whole-provider budget.
                self._by_provider[provider] = float(amount)
            elif not model:
                continue
            elif provider in (None, "*", ""):
                self._by_model[model] = float(amount)
            else:
                self._by_pair[(provider, model)] = float(amount)

    def get(self, provider: str, model: str) -> float | None:
        """The cap for a pair, or ``None`` when none is declared."""
        if (found := self._by_pair.get((provider, model))) is not None:
            return found
        if (found := self._by_model.get(model)) is not None:
            return found
        return self._by_provider.get(provider)

    def __bool__(self) -> bool:
        return bool(self._by_pair or self._by_model or self._by_provider)

    def __len__(self) -> int:
        return len(self._by_pair) + len(self._by_model) + len(self._by_provider)


class SpendLedger:
    """Estimated spend and call counts per ``(provider, model)``.

    One lock for the whole ledger. Writes come from the request path and reads
    from the routing path, and both are short dict operations, so a coarse lock
    is the right shape here: it cannot deadlock because nothing is held across
    an ``await``, and contention is a dict update either way.
    """

    def __init__(self, billing: Any = None, *, priced: bool = False) -> None:
        self._billing = billing
        # Whether `[telemetry.costs]` actually holds prices. Metering needs
        # this *and* a cycle; see `metering`.
        self._priced = bool(priced)
        self.budgets = BudgetTable(getattr(billing, "budgets", ()) or ())
        self._spend: dict[tuple[str, str], float] = {}
        self._calls: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        # The window these totals belong to. Comparing it on every access is
        # what makes the rollover automatic: a new window is a different value,
        # so the old totals are dropped instead of subtracting from a budget
        # that has just been topped up.
        self._window_start: float | None = None

    @property
    def metering(self) -> bool:
        """Whether spend can be *metered* -- i.e. costed against a cycle.

        True only when **both** halves are present: prices in
        ``[telemetry.costs]`` and a billing cycle (``reset_day``, or the
        explicit period). Prices alone cannot say what was spent *this month*;
        a cycle alone divides by an unknown. With either missing the total is
        not a number anyone should route on, so ``highest_balance`` treats
        metering as off and falls back to ``least_used`` rather than ranking on
        a figure it cannot stand behind.

        Note this is independent of whether a *budget* is declared. Metering
        being on just means the spend figure is trustworthy; a candidate still
        needs its own cap to have a balance at all.
        """
        return self._priced and self._window(None) is not None

    # -- recording ----------------------------------------------------------

    def record(
        self,
        provider: str | None,
        model: str | None,
        cost_usd: float | None = None,
        *,
        at: float | None = None,
    ) -> None:
        """Add one call to the ledger.

        Call count is recorded even when the cost is unknown, because
        ``least_used`` ranks on calls and does not need a price table at all.
        A record outside the current window is dropped rather than counted --
        otherwise spend from a rolled-over month would keep subtracting from
        the new month's budget.
        """
        if not provider or not model:
            return
        if not self._in_window(at):
            return
        with self._lock:
            self._roll(at)
            key = (provider, model)
            self._calls[key] = self._calls.get(key, 0) + 1
            if cost_usd:
                self._spend[key] = self._spend.get(key, 0.0) + float(cost_usd)

    def reset(self) -> None:
        """Forget everything. Called when the billing window rolls over."""
        with self._lock:
            self._spend.clear()
            self._calls.clear()

    def _roll(self, at: float | None) -> None:
        """Clear totals when the billing window has moved on. Caller holds the lock."""
        window = self._window(at)
        start = window[0] if window is not None else None
        if start != self._window_start:
            self._spend.clear()
            self._calls.clear()
            self._window_start = start

    # -- reading ------------------------------------------------------------

    def spend(self, provider: str, model: str, *, at: float | None = None) -> float:
        """Estimated ``$`` spent in the current window."""
        with self._lock:
            self._roll(at)
            return self._spend.get((provider, model), 0.0)

    def calls(self, provider: str, model: str, *, at: float | None = None) -> int:
        """Calls made in the current window."""
        with self._lock:
            self._roll(at)
            return self._calls.get((provider, model), 0)

    def balance(self, provider: str, model: str, *, at: float | None = None) -> float | None:
        """Declared cap minus estimated spend, or ``None`` if no cap exists.

        ``None`` means "nothing was declared" and the caller decides what that
        is worth -- ``highest_balance`` reads it as *immaterial* and treats it
        as a neutral zero, so an undeclared model neither outranks a funded one
        nor is buried behind an exhausted one.
        """
        cap = self.budgets.get(provider, model)
        if cap is None:
            return None
        return cap - self.spend(provider, model, at=at)

    def total_spend(self, *, at: float | None = None) -> float:
        with self._lock:
            self._roll(at)
            return sum(self._spend.values())

    def describe(self, *, at: float | None = None) -> dict[str, Any]:
        """Introspection payload for ``/v1/margAI/tags`` and the doctor."""
        billing = self._billing
        window = self._window(at)
        with self._lock:
            self._roll(at)
            spend = {f"{p}/{m}": round(v, 6) for (p, m), v in sorted(self._spend.items())}
            calls = {f"{p}/{m}": n for (p, m), n in sorted(self._calls.items())}
            total = sum(self._spend.values())
        return {
            "reset_day": getattr(billing, "reset_day", None),
            "window": (
                None
                if window is None
                else {
                    "start": window[0],
                    "end": window[1],
                    "current": billing.is_current(at),
                }
            ),
            "budgets": len(self.budgets),
            "spend": spend,
            "calls": calls,
            "total_spend": round(total, 6),
            "process_local": True,
        }

    # -- window -------------------------------------------------------------

    def _window(self, at: float | None) -> tuple[float, float] | None:
        return None if self._billing is None else self._billing.window(at)

    def _in_window(self, at: float | None = None) -> bool:
        """Whether a timestamp belongs to the current billing window.

        True when no window is configured: the ledger still tracks call counts
        (``least_used`` needs no billing setup at all), and the bounds that
        exist to stop last month's spend corrupting this month's budget do not
        apply when there is no budget.
        """
        window = self._window(at)
        if window is None:
            return True
        current = time.time() if at is None else at
        return window[0] <= current <= window[1]
