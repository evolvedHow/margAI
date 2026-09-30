"""Virtual models: stable names in front of a changing catalogue.

A client that hardcodes ``margAI/openai/gpt-4o`` has made two promises: that
this gateway hosts it, and that it stays there. A client that sends
``margAI/fast`` has made one. The gateway resolves the second kind per call
against the models actually configured right now.

Each virtual model is a :class:`ModelPolicy` from ``config`` -- a set of
filters over candidates plus one strategy for picking among the survivors.
The division matters: filters are a *function of the catalogue and the call*,
so they are checkable and testable, while the strategy is the only part that
may look at history (round-robin, least-used, highest-balance) and is the
reason a route needs a ``reason`` field to say what happened.

The design rule is that a policy can only ever *narrow* the field. A caller
who writes ``!not: openai`` is excluded from the answer no matter what any
policy says, so an operator cannot accidentally re-expose a provider someone
deliberately opted out of on this call. The caller's constraints are applied
last and are absolute.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

from .billing import SpendLedger
from .errors import ApiError
from .router import Candidate, Route

__all__ = ["CostTable", "VirtualModel", "VirtualModels"]

logger = logging.getLogger("margAI.virtual")


class CostTable:
    """Prices keyed ``provider/model``, for ranking by cost.

    Built from the same ``[telemetry.costs]`` table the telemetry layer uses,
    so a cost ceiling and a cost report can never disagree about a price.

    An entry keyed only by model name (no ``provider/`` prefix) applies to any
    provider offering that model, which is what makes a price table survive a
    provider rename. The most specific match wins, so an explicit
    ``provider/model`` row overrides a bare-model row.
    """

    def __init__(self, entries: Iterable[Any] = ()) -> None:
        self._by_pair: dict[tuple[str, str], float] = {}
        self._by_model: dict[str, float] = {}
        for entry in entries:
            provider = getattr(entry, "provider", None)
            model = getattr(entry, "model", None)
            if not model:
                continue
            price = blended(getattr(entry, "input_price", 0.0), getattr(entry, "output_price", 0.0))
            if provider in (None, "*", ""):
                self._by_model[model] = price
            else:
                self._by_pair[(provider, model)] = price

    def get(self, provider: str, model: str) -> float | None:
        """Blended $/1M for a pair, or ``None`` when nothing is known."""
        if (found := self._by_pair.get((provider, model))) is not None:
            return found
        return self._by_model.get(model)

    def __bool__(self) -> bool:
        return bool(self._by_pair or self._by_model)


def blended(input_price: float, output_price: float) -> float:
    """A 3:1 input:output ratio.

    The choice is arbitrary but it must be *fixed*: a "cheapest" strategy whose
    ranking depended on a ratio the operator never chose would be a ranking
    they could not reason about. 3:1 is roughly a typical chat turn.
    """
    return (float(input_price) * 3.0 + float(output_price)) / 4.0


class VirtualModel:
    """One configured policy, bound to a name and a router."""

    def __init__(
        self,
        policy: Any,
        prefix: str,
        costs: CostTable | None = None,
        ledger: SpendLedger | None = None,
    ) -> None:
        self.policy = policy
        self.prefix = prefix
        self.name = policy.name
        self.strategy = policy.strategy
        self.costs = costs if costs is not None else CostTable()
        # History for the two strategies that need it. The ledger is shared by
        # every virtual model on the gateway, because spend and call counts are
        # properties of a *model*, not of the policy that happened to ask for
        # it -- one policy's traffic must count against another's budget.
        self.ledger = ledger
        # One cursor for the whole virtual model, shared by every call, behind
        # one lock. Round-robin that kept a per-request cursor would restart
        # the rotation on each call and send every request to the same model --
        # load spreading that provably does not spread load.
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """The client-facing id, e.g. ``margAI/fast``."""
        return self.policy.id(self.prefix)

    def applies_to(self, intent: Any) -> bool:
        """True when this call opted in to the policy.

        A policy gated on tags is the opt-in form: ``!fast:`` on a call turns
        the virtual model on, and no tag means the operator's default does not
        silently re-route traffic.
        """
        required = self.policy.tags
        if not required:
            return True
        return all(intent.wants(tag) for tag in required)

    def eligible(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        """Apply the policy's filters, preserving candidate order."""
        policy = self.policy
        out: list[Candidate] = []
        for cand in candidates:
            if policy.providers and cand.provider not in policy.providers:
                continue
            if cand.provider in policy.exclude_providers:
                continue
            if policy.models and not _matches_any(policy.models, cand):
                continue
            if policy.max_cost_per_1m is not None:
                price = self.costs.get(cand.provider, cand.model)
                # Unknown price cannot be shown to be under a ceiling, so it is
                # excluded. Guessing here would make a cost limit decorative.
                if price is None or price > policy.max_cost_per_1m:
                    continue
            out.append(cand)
        return out

    def _preferred(self, candidates: list[Candidate]) -> list[Candidate]:
        """Reorder so ``prefer`` entries come first, in the declared order.

        A preference that matches nothing is skipped, not fatal: preferences
        name models an operator likes, and models get retired.
        """
        if not self.policy.prefer:
            return candidates
        pool = list(candidates)
        front: list[Candidate] = []
        for want in self.policy.prefer:
            for idx, cand in enumerate(pool):
                if _matches(cand, want):
                    front.append(pool.pop(idx))
                    break
        return front + pool

    def choose(self, candidates: list[Candidate]) -> tuple[Candidate, str] | None:
        """Pick one candidate. Returns ``None`` when there is nothing to pick.

        ``(candidate, reason)`` -- the reason is appended to the Route so an
        operator reading logs can see which strategy decided, rather than
        inferring it from a model id.
        """
        pool = self._preferred(candidates)
        if not pool:
            return None
        if self.strategy == "first":
            return pool[0], f"virtual:{self.name}/first"
        if self.strategy == "cheapest":
            return _cheapest(pool, self.costs), f"virtual:{self.name}/cheapest"
        if self.strategy == "round_robin":
            return self._rotate(pool), f"virtual:{self.name}/round_robin"
        if self.strategy == "least_used":
            return _least_used(pool, self.ledger, self.name), f"virtual:{self.name}/least_used"
        if self.strategy == "highest_balance":
            return (
                _highest_balance(pool, self.ledger, self.name),
                f"virtual:{self.name}/highest_balance",
            )
        return pool[0], f"virtual:{self.name}/{self.strategy}"

    def _rotate(self, pool: list[Candidate]) -> Candidate:
        with self._lock:
            picked = self._cursor % len(pool)
            self._cursor = (self._cursor + 1) % len(pool)
        return pool[picked]

    def describe(self) -> dict[str, Any]:
        """Introspection payload for /v1/margAI/tags and the doctor."""
        policy = self.policy
        return {
            "id": self.model_id,
            "name": self.name,
            "strategy": self.strategy,
            "tags": list(policy.tags),
            "providers": list(policy.providers),
            "exclude_providers": list(policy.exclude_providers),
            "models": list(policy.models),
            "prefer": list(policy.prefer),
            "max_cost_per_1m": policy.max_cost_per_1m,
        }


def _cheapest(pool: list[Candidate], costs: CostTable) -> Candidate:
    """The lowest-price candidate; unknown prices sort last, ties break on order.

    Sorting unknown prices last rather than dropping them means a policy with a
    ``cheapest`` strategy still works on a gateway with no price table at all --
    it just falls back to the first candidate instead of failing to route.

    The fallback is deliberate but silent, which is the one thing a ``cheapest``
    policy should never be: an operator who added prices and mistyped a model id
    gets first-candidate ordering with no sign the ranking did nothing. So the
    two paths that degrade to ordering-by-position say so in the log. ``debug``
    rather than ``warning`` because an absent price table is a valid
    configuration, not a mistake -- but it is the one case where a policy's
    advertised behavior is not what it did.
    """
    if not costs:
        logger.debug(
            "virtual model 'cheapest': [telemetry.costs] is empty, so ranking "
            "fell back to the first candidate"
        )
        return pool[0]
    if not any(costs.get(c.provider, c.model) is not None for c in pool):
        logger.debug(
            "virtual model 'cheapest': none of the %d candidates are priced in "
            "[telemetry.costs], so ranking fell back to the first candidate",
            len(pool),
        )
    return min(pool, key=lambda c: _price_key(costs.get(c.provider, c.model)))


def _price_key(price: float | None) -> tuple[bool, float]:
    """Sort key putting unknown prices after every known one."""
    return (price is None, price if price is not None else 0.0)


def _least_used(pool: list[Candidate], ledger: SpendLedger | None, name: str) -> Candidate:
    """The candidate with the fewest calls this billing window.

    Ties fall to candidate order, so the result is deterministic and a policy
    with no traffic spread at all still resolves to the same model every time.
    Counts are per model, not per policy, so traffic through a *different*
    virtual model counts here too.
    """
    if ledger is None:
        logger.debug(
            "virtual model '%s': least_used has no ledger and fell back to the "
            "first candidate",
            name,
        )
        return pool[0]
    return min(pool, key=lambda c: ledger.calls(c.provider, c.model))


def _highest_balance(pool: list[Candidate], ledger: SpendLedger | None, name: str) -> Candidate:
    """The candidate with the most budget left, then the fewest calls.

    Metering -- and so this strategy -- needs *both* ``[telemetry.costs]`` and a
    billing cycle. With either missing the spend figure is not one to route on,
    so this falls back to :func:`_least_used` rather than ranking on a number it
    cannot stand behind. ``highest_balance`` on a gateway that never configured
    billing therefore degrades to ``least_used``, which is the useful behavior
    rather than an error.

    A candidate with **no declared budget is immaterial**, exactly as if it had
    spent its way to a zero balance: it is neither rewarded for an unspendable
    cap nor punished for an unmeasurable one. That keeps an unpriced local model
    and a proprietary model whose spend is tracked elsewhere in the running
    without letting either distort the comparison -- a candidate with real room
    left (positive balance) still wins, and an exhausted one (negative) still
    loses to it. Ties break on call count, then candidate order.
    """
    if ledger is None or not ledger.metering:
        logger.debug(
            "virtual model '%s': highest_balance needs prices in [telemetry.costs] "
            "and a billing cycle (reset_day or a period); falling back to least_used",
            name,
        )
        return _least_used(pool, ledger, name)
    if not ledger.budgets:
        logger.debug(
            "virtual model '%s': highest_balance has no [telemetry.budgets] to rank "
            "against; falling back to least_used",
            name,
        )
        return _least_used(pool, ledger, name)

    def key(cand: Candidate) -> tuple[float, int]:
        remaining = ledger.balance(cand.provider, cand.model)
        # None == no declared cap == immaterial == neutral zero.
        balance = 0.0 if remaining is None else remaining
        return (-balance, ledger.calls(cand.provider, cand.model))

    return min(pool, key=key)


def _matches(cand: Candidate, want: str) -> bool:
    if "/" in want:
        provider, _, model = want.partition("/")
        return cand.provider == provider and cand.model == model
    return cand.model == want


def _matches_any(wants: tuple[str, ...], cand: Candidate) -> bool:
    return any(_matches(cand, want) for want in wants)


class VirtualModels:
    """The virtual models configured for one gateway."""

    def __init__(
        self,
        prefix: str,
        policies: Iterable[Any] = (),
        costs: CostTable | None = None,
        ledger: SpendLedger | None = None,
    ) -> None:
        self.prefix = prefix
        self.costs = costs if costs is not None else CostTable()
        self.ledger = ledger
        self._models: dict[str, VirtualModel] = {}
        for policy in policies:
            self._models[policy.name] = VirtualModel(policy, prefix, self.costs, ledger)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._models

    def get(self, name: str) -> VirtualModel | None:
        return self._models.get(name)

    def policies(self) -> dict[str, Any]:
        """The configured policies, by name.

        Exposed so a per-call overlay can start from the deployment's set and
        override one entry, instead of re-declaring every model to change one.
        """
        return {name: model.policy for name, model in self._models.items()}

    def ids(self) -> list[str]:
        """Client-facing ids, e.g. ``margAI/fast``."""
        return [model.model_id for model in self._models.values()]

    def name_for(self, model_id: str) -> str | None:
        """The policy name a client id names, or ``None``."""
        prefix = self.prefix + "/"
        if not model_id.startswith(prefix):
            return None
        rest = model_id[len(prefix) :]
        # A two-segment id is a virtual model; a three-segment one names a
        # provider, which is a different namespace entirely.
        if "/" in rest:
            return None
        return rest if rest in self._models else None

    def is_virtual(self, model_id: str | None) -> bool:
        return self.name_for(model_id or "") is not None

    def check_collisions(self, provider_names: Iterable[str], model_ids: Iterable[str]) -> None:
        """Reject a virtual name that shadows a real provider or model.

        Checked once at load, not per call. A collision would otherwise be
        resolved by whichever code path happened to run first, which is the
        kind of ambiguity that is invisible until it routes a request to the
        wrong place in production.
        """
        providers = set(provider_names)
        models = set(model_ids)
        for name in self._models:
            if name in providers:
                raise ValueError(
                    f"virtual model '{name}' collides with a configured provider of the same name; "
                    "rename the provider or the virtual model"
                )
            if name in models:
                raise ValueError(
                    f"virtual model '{name}' collides with a real model id; "
                    f"the client id '{self.prefix}/{name}' would be ambiguous. Rename one of them."
                )

    def resolve(self, model_id: str | None, intent: Any, candidates: Iterable[Candidate]) -> Route | None:
        """Resolve a virtual model id against the candidates.

        Returns ``None`` when the id is not a virtual model, or when the policy
        is not applicable and no other policy claims the call. Raises when the
        id *is* a virtual model but nothing satisfies it, because the caller
        asked for a specific thing and silence would be a wrong answer.
        """
        name = self.name_for(model_id or "")
        if name is None:
            return None
        model = self._models[name]
        if not model.applies_to(intent):
            raise ApiError(
                400,
                f"Virtual model '{model_id}' requires tag(s) "
                f"{list(model.policy.tags)} on this call.",
                error_type="invalid_request_error",
                param="model",
            )
        pool = model.eligible(candidates)
        picked = model.choose(pool)
        if picked is None:
            raise self.explain(model, len(list(candidates)))
        cand, reason = picked
        return Route(cand.provider, cand.model, reason)

    def explain(self, model: VirtualModel, pool_size: int) -> ApiError:
        """The error for a virtual model nothing satisfies."""
        policy = model.policy
        if pool_size == 0:
            detail = "no models are configured on this gateway"
        elif policy.max_cost_per_1m is not None and not self.costs:
            detail = (
                f"max_cost_per_1m = {policy.max_cost_per_1m} is set but no prices are "
                "configured in [telemetry.costs], so no candidate can be shown to qualify"
            )
        elif policy.max_cost_per_1m is not None:
            detail = (
                f"no configured model is priced at or below "
                f"{policy.max_cost_per_1m}/1M; the cheapest known candidates were excluded"
            )
        else:
            detail = "its filters exclude every configured model"
        return ApiError(
            404,
            f"Virtual model '{model.model_id}' matched no candidate: {detail}. "
            f"See /v1/margAI/tags for the policy.",
            error_type="invalid_request_error",
            param="model",
        )

    def describe(self) -> list[dict[str, Any]]:
        return [model.describe() for model in self._models.values()]
