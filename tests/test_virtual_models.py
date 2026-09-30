"""Virtual models: names that promise a policy instead of a model id.

The property under test throughout is that a policy can only *narrow* the
field. A caller who excluded a provider must not get it back because some
operator-written virtual model happened to prefer it.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from conftest import FakeTransport, make_wrapper, provider_config

from margAI.config import BillingConfig, CostEntry, ModelPolicy
from margAI.core.billing import SpendLedger
from margAI.core.context import RequestContext
from margAI.core.marglets import ROUTE_REASON_KEY
from margAI.core.router import Candidate
from margAI.core.virtual import CostTable, VirtualModel, VirtualModels

PROVIDERS = (
    provider_config("openai", models=("gpt-4o", "gpt-4o-mini")),
    provider_config("groq", models=("llama-3.3-70b",), default_model="llama-3.3-70b"),
    provider_config("local", models=("qwen2.5-coder",), default_model="qwen2.5-coder"),
)

# Prices chosen so cost order is unambiguous and differs from alphabetical
# order: local is free, openai's mini is cheap, openai's flagship is dear.
COSTS = (
    CostEntry("openai", "gpt-4o", 2.50, 10.00),
    CostEntry("openai", "gpt-4o-mini", 0.15, 0.60),
    CostEntry("groq", "llama-3.3-70b", 0.59, 0.79),
)


def build(*policies: ModelPolicy, costs: tuple = COSTS, providers=PROVIDERS, billing=None):
    return make_wrapper(
        FakeTransport(),
        providers=providers,
        models={p.name: p for p in policies},
        costs=costs,
        billing=billing,
    )


def call(app, model: str, *tags: str):
    """Make one call. Returns ``(response, ctx)``, success or not.

    A `before` hook at order=-40 runs early enough to capture the context
    object, and the router writes the route into that same object's state
    afterwards -- so the context read after the call finishes is the one the
    call actually ran with.
    """
    captured: list[RequestContext] = []

    @app.before(order=-40)
    def capture(ctx: RequestContext) -> None:
        captured.append(ctx)

    directive = ("!margAI: " + " ".join(tags) + "\n") if tags else ""
    body = {"model": model, "messages": [{"role": "user", "content": directive + "hi"}]}
    response = asyncio.run(app.complete(body))
    return response, (captured[-1] if captured else None)


def ask(app, model: str, *tags: str) -> RequestContext:
    """A call that must succeed, handing back the context it ran with."""
    response, ctx = call(app, model, *tags)
    assert response.status == 200, response.body
    assert ctx is not None
    return ctx


def refused(app, model: str, *tags: str) -> dict:
    """A call that must be refused, handing back the error body.

    `complete` turns an ApiError into a response rather than raising, which is
    what a client sees -- so that is what these assert on.
    """
    response, _ctx = call(app, model, *tags)
    assert response.status >= 400, f"expected a refusal, got {response.status}: {response.body}"
    return response.body


def upstream_model(app, model: str = "margAI/fast", *tags: str) -> str:
    """The model id that actually reached the wire.

    Asserted on the prepared request rather than on anything the wrapper
    reports, because "what the policy decided" and "what the provider was
    sent" are exactly the two things a bug would let drift apart.
    """
    ask(app, model, *tags)
    return app.transport.requested[-1].json["model"]


# -- resolution basics ------------------------------------------------------


def test_virtual_model_resolves_to_a_real_target():
    app = build(ModelPolicy(name="fast", strategy="first"))
    assert upstream_model(app) == "gpt-4o"


def test_a_non_virtual_id_is_still_a_plain_lookup():
    app = build(ModelPolicy(name="fast"))
    assert upstream_model(app, "margAI/local/qwen2.5-coder") == "qwen2.5-coder"


def test_a_three_segment_id_is_never_a_virtual_model():
    """A virtual name is exactly one segment, so `margAI/openai/gpt-4o` stays
    a provider lookup however temptingly its middle segment looks."""
    virtuals = VirtualModels("margAI", [ModelPolicy(name="openai")])
    assert virtuals.name_for("margAI/openai/gpt-4o") is None
    assert virtuals.name_for("margAI/openai") == "openai"


def test_an_unknown_virtual_name_is_a_normal_404():
    app = build(ModelPolicy(name="fast"))
    assert "Unknown model" in refused(app, "margAI/nonexistent")["error"]["message"]


# -- filters ----------------------------------------------------------------


def test_provider_allow_list():
    app = build(ModelPolicy(name="groqonly", providers=("groq",)))
    assert upstream_model(app, "margAI/groqonly") == "llama-3.3-70b"


def test_provider_deny_list_is_applied_after_the_allow_list():
    app = build(
        ModelPolicy(name="narrow", providers=("openai", "groq"), exclude_providers=("openai",))
    )
    assert upstream_model(app, "margAI/narrow") == "llama-3.3-70b"


def test_model_allow_list_accepts_bare_and_qualified_forms():
    app = build(ModelPolicy(name="minis", models=("gpt-4o-mini", "groq/llama-3.3-70b")))
    assert upstream_model(app, "margAI/minis") == "gpt-4o-mini"


def test_cost_ceiling_excludes_the_expensive_models():
    app = build(ModelPolicy(name="cheap", strategy="first", max_cost_per_1m=1.0))
    # gpt-4o blends to 4.375, so it is out; the mini blends to 0.2625.
    assert upstream_model(app, "margAI/cheap") == "gpt-4o-mini"


def test_a_cost_ceiling_with_no_price_table_fails_loudly():
    """An unknown price cannot be shown to be under a ceiling, so a policy that
    silently routed anyway would make the ceiling decorative."""
    app = build(ModelPolicy(name="cheap", max_cost_per_1m=1.0), costs=())
    assert "telemetry.costs" in refused(app, "margAI/cheap")["error"]["message"]


def test_prefer_orders_the_field_without_removing_anything():
    app = build(ModelPolicy(name="picky", prefer=("groq/llama-3.3-70b",)))
    assert upstream_model(app, "margAI/picky") == "llama-3.3-70b"


def test_a_preference_that_matches_nothing_is_skipped_not_fatal():
    """Preferences name models an operator likes; models get retired."""
    app = build(ModelPolicy(name="picky", prefer=("openai/gpt-5-ultra", "local/qwen2.5-coder")))
    assert upstream_model(app, "margAI/picky") == "qwen2.5-coder"


# -- strategies -------------------------------------------------------------


def test_cheapest_ranks_by_blended_price():
    app = build(ModelPolicy(name="thrifty", strategy="cheapest"))
    # The local model has no price, so it sorts last rather than winning by
    # default; the mini is the cheapest known at 0.2625.
    assert upstream_model(app, "margAI/thrifty") == "gpt-4o-mini"


def test_cheapest_falls_back_to_first_without_any_prices():
    app = build(ModelPolicy(name="thrifty", strategy="cheapest"), costs=())
    assert upstream_model(app, "margAI/thrifty") == "gpt-4o"


def test_cheapest_says_so_when_it_falls_back(caplog: pytest.LogCaptureFixture):
    """The fallback is deliberate, so it must not be silent.

    An operator who added a cost table and mistyped a model id gets
    first-candidate ordering with no sign the ranking did nothing. Debug rather
    than warning: no price table is a valid configuration, just not the one
    that makes `cheapest` mean anything.
    """
    app = build(ModelPolicy(name="thrifty", strategy="cheapest"), costs=())
    with caplog.at_level(logging.DEBUG, logger="margAI.virtual"):
        assert upstream_model(app, "margAI/thrifty") == "gpt-4o"
    assert "telemetry.costs" in caplog.text


def test_cheapest_stays_quiet_when_it_actually_ranks(caplog: pytest.LogCaptureFixture):
    """The other half: a policy that ranked on real prices must not log the
    fallback, or the diagnostic is noise an operator learns to ignore."""
    app = build(ModelPolicy(name="thrifty", strategy="cheapest"))
    with caplog.at_level(logging.DEBUG, logger="margAI.virtual"):
        assert upstream_model(app, "margAI/thrifty") == "gpt-4o-mini"
    assert "telemetry.costs" not in caplog.text


def test_first_is_deterministic_across_calls():
    app = build(ModelPolicy(name="steady", strategy="first"))
    assert [upstream_model(app, "margAI/steady") for _ in range(5)] == ["gpt-4o"] * 5


def test_round_robin_cycles_through_the_whole_field():
    app = build(ModelPolicy(name="rr", strategy="round_robin", providers=("openai",)))
    picks = [upstream_model(app, "margAI/rr") for _ in range(4)]
    assert picks == ["gpt-4o", "gpt-4o-mini", "gpt-4o", "gpt-4o-mini"]


def test_round_robin_keeps_one_cursor_across_threads():
    """A per-request cursor would restart the rotation on every call and send
    every request to the same model: load spreading that does not spread.

    Asserted on the policy object rather than through the gateway, because
    reading "which model did *my* call send" back off a shared transport is
    itself racy -- the property under test is about the cursor, and the cursor
    is what has to be shared.
    """
    model = VirtualModels("margAI", [ModelPolicy(name="rr", strategy="round_robin")]).get("rr")
    assert model is not None
    pool = [Candidate("openai", "gpt-4o"), Candidate("openai", "gpt-4o-mini")]
    seen: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(25):
            picked, _reason = model.choose(pool)  # type: ignore[misc]
            with lock:
                seen.append(picked.model)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 200 picks over 2 candidates, and the cursor is a single counter under a
    # single lock, so the split can only be off by the number of threads.
    assert 95 <= seen.count("gpt-4o") <= 105, seen.count("gpt-4o")


# -- history-based strategies -----------------------------------------------


def ledger_for(budgets: tuple = (), *, priced: bool = True, reset_day: int | None = 1):
    return SpendLedger(BillingConfig(reset_day=reset_day, budgets=budgets), priced=priced)


def pick(strategy: str, pool: list[Candidate], ledger: SpendLedger | None) -> str:
    model = VirtualModel(ModelPolicy(name="p", strategy=strategy), "margAI", CostTable(), ledger)
    picked, _reason = model.choose(pool)  # type: ignore[misc]
    return picked.model


POOL = [Candidate("openai", "gpt-4o"), Candidate("groq", "llama-3.3-70b")]


def test_least_used_picks_the_fewest_calls():
    ledger = ledger_for()
    ledger.record("openai", "gpt-4o")
    ledger.record("openai", "gpt-4o")
    assert pick("least_used", POOL, ledger) == "llama-3.3-70b"


def test_least_used_ties_fall_to_candidate_order():
    ledger = ledger_for()
    assert pick("least_used", POOL, ledger) == "gpt-4o"


def test_highest_balance_picks_the_most_remaining():
    ledger = ledger_for((CostEntry("openai", "gpt-4o", 10.0, 0.0), CostEntry("groq", "llama-3.3-70b", 10.0, 0.0)))
    ledger.record("openai", "gpt-4o", 9.0)  # $1 left
    ledger.record("groq", "llama-3.3-70b", 1.0)  # $9 left
    assert pick("highest_balance", POOL, ledger) == "llama-3.3-70b"


def test_highest_balance_prefers_a_funded_candidate_over_an_unbudgeted_one():
    """No declared budget is immaterial: a neutral zero, not a free pass."""
    ledger = ledger_for((CostEntry("openai", "gpt-4o", 10.0, 0.0),))
    ledger.record("openai", "gpt-4o", 5.0)  # balance +5 > 0, so it wins
    assert pick("highest_balance", POOL, ledger) == "gpt-4o"


def test_an_exhausted_budget_loses_to_an_unbudgeted_candidate():
    ledger = ledger_for((CostEntry("openai", "gpt-4o", 10.0, 0.0),))
    ledger.record("openai", "gpt-4o", 25.0)  # balance -15 < 0, so it loses
    assert pick("highest_balance", POOL, ledger) == "llama-3.3-70b"


def test_highest_balance_falls_back_to_least_used_without_prices():
    """Prices without a cycle cannot meter either; both halves are required."""
    ledger = ledger_for((CostEntry("openai", "gpt-4o", 10.0, 0.0),), reset_day=None)
    ledger.record("openai", "gpt-4o")
    ledger.record("openai", "gpt-4o")
    assert pick("highest_balance", POOL, ledger) == "llama-3.3-70b"


def test_highest_balance_falls_back_when_no_budgets_are_declared():
    ledger = ledger_for()
    ledger.record("openai", "gpt-4o")
    assert pick("highest_balance", POOL, ledger) == "llama-3.3-70b"


def test_least_used_without_a_ledger_falls_back_to_the_first():
    assert pick("least_used", POOL, None) == "gpt-4o"


def test_a_finished_call_feeds_the_gateway_ledger():
    """Spend and call counts are recorded on the way out, per model."""
    ledger_billing = BillingConfig(
        reset_day=1, budgets=(CostEntry("openai", "gpt-4o", 10.0, 0.0),)
    )
    app = build(
        ModelPolicy(name="fast", strategy="highest_balance", providers=("openai",)),
        billing=ledger_billing,
    )
    ask(app, "margAI/fast")
    ask(app, "margAI/fast")
    assert app.ledger.calls("openai", "gpt-4o") == 2


# -- the narrowing invariant ------------------------------------------------


def test_a_caller_exclusion_beats_a_policy_preference():
    """The whole point: a policy can narrow the field, never widen it."""
    app = build(ModelPolicy(name="picky", prefer=("groq/llama-3.3-70b",)))
    assert upstream_model(app, "margAI/picky", "not=groq") == "gpt-4o"


def test_a_caller_exclusion_can_empty_the_field_and_says_so():
    app = build(ModelPolicy(name="picky", providers=("groq",)))
    assert "matched no candidate" in refused(app, "margAI/picky", "not=groq")["error"]["message"]


def test_the_caller_filter_runs_before_the_strategy():
    """Round-robin must not hand out slots to targets this call excluded --
    otherwise the rotation spends positions on candidates nobody can receive."""
    app = build(ModelPolicy(name="rr", strategy="round_robin"))
    picks = [upstream_model(app, "margAI/rr", "not=openai") for _ in range(3)]
    assert picks == ["llama-3.3-70b", "qwen2.5-coder", "llama-3.3-70b"]


# -- tag-gated policies -----------------------------------------------------


def test_a_tag_gated_policy_only_applies_to_opted_in_calls():
    app = build(ModelPolicy(name="turbo", tags=("turbo",), strategy="first"))
    assert "requires tag" in refused(app, "margAI/turbo")["error"]["message"]


def test_a_tag_gated_policy_resolves_once_the_tag_is_present():
    app = build(ModelPolicy(name="turbo", tags=("turbo",), providers=("groq",)))
    assert upstream_model(app, "margAI/turbo", "turbo") == "llama-3.3-70b"


# -- a bangtag pin beats a policy -------------------------------------------


def test_a_concrete_bangtag_route_overrides_a_virtual_model():
    app = build(ModelPolicy(name="fast", strategy="first"))
    assert upstream_model(app, "margAI/fast", "route=local/qwen2.5-coder") == "qwen2.5-coder"


# -- collision checks at load time -----------------------------------------


def test_a_virtual_name_may_not_shadow_a_provider():
    with pytest.raises(ValueError, match="collides with a configured provider"):
        make_wrapper(FakeTransport(), providers=PROVIDERS, models={"openai": ModelPolicy(name="openai")})


def test_a_virtual_name_may_not_shadow_a_real_model_id():
    with pytest.raises(ValueError, match="collides with a real model id"):
        make_wrapper(
            FakeTransport(),
            providers=PROVIDERS,
            models={"gpt-4o": ModelPolicy(name="gpt-4o")},
        )


def test_a_real_model_id_is_still_reachable_when_no_policy_shadows_it():
    app = build(ModelPolicy(name="fast"))
    assert upstream_model(app, "margAI/openai/gpt-4o") == "gpt-4o"


# -- cost table -------------------------------------------------------------


def test_an_explicit_pair_price_beats_a_bare_model_price():
    table = CostTable(
        (
            CostEntry("*", "gpt-4o", 1.0, 1.0),
            CostEntry("groq", "gpt-4o", 9.0, 9.0),
        )
    )
    assert table.get("openai", "gpt-4o") == blended_of(1.0, 1.0)
    assert table.get("groq", "gpt-4o") == blended_of(9.0, 9.0)


def test_an_unknown_price_is_reported_as_unknown_not_zero():
    assert CostTable(COSTS).get("local", "qwen2.5-coder") is None


def blended_of(inp: float, out: float) -> float:
    return (inp * 3.0 + out) / 4.0


# -- introspection ----------------------------------------------------------


def test_ids_and_describe_expose_the_policy_for_introspection():
    app = build(ModelPolicy(name="fast", strategy="cheapest", max_cost_per_1m=1.0, exclude_providers=("local",)))
    assert app.virtuals.ids() == ["margAI/fast"]
    (entry,) = app.virtuals.describe()
    assert entry == {
        "id": "margAI/fast",
        "name": "fast",
        "strategy": "cheapest",
        "tags": [],
        "providers": [],
        "exclude_providers": ["local"],
        "models": [],
        "prefer": [],
        "max_cost_per_1m": 1.0,
    }


def test_an_empty_gateway_reports_no_virtual_models():
    app = make_wrapper(FakeTransport())
    assert app.virtuals.ids() == []
    assert len(app.virtuals) == 0


# -- end to end through the wire -------------------------------------------


def test_a_virtual_model_reaches_the_upstream_with_the_resolved_id():
    app = build(ModelPolicy(name="groqfast", providers=("groq",)))
    ctx = ask(app, "margAI/groqfast")
    assert app.transport.requested[-1].json["model"] == "llama-3.3-70b"
    assert ctx.state[ROUTE_REASON_KEY] == "virtual:groqfast/first"


def test_the_route_reason_names_the_deciding_strategy():
    """An operator reading logs should be able to see *why* a model was
    chosen rather than inferring it from the id."""
    app = build(ModelPolicy(name="cheap", strategy="cheapest", max_cost_per_1m=1.0))
    ctx = ask(app, "margAI/cheap")
    assert ctx.state[ROUTE_REASON_KEY] == "virtual:cheap/cheapest"
