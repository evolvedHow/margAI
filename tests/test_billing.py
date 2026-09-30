"""The spend ledger and the billing-cycle arithmetic behind it.

Two invariants run through this file. First, a window is *recurring*: it rolls
forward on its own, so a gateway left running for months never needs a config
edit and never goes stale. Second, metering is on only when **both** prices and
a cycle are configured -- with either missing the strategy falls back rather
than routing on a figure it cannot stand behind.
"""

from __future__ import annotations

from datetime import UTC, datetime

from margAI.config import BillingConfig, CostEntry
from margAI.core.billing import BudgetTable, SpendLedger


def at(year: int, month: int, day: int, hour: int = 12) -> float:
    return datetime(year, month, day, hour, tzinfo=UTC).timestamp()


def day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).date().isoformat()


# -- window arithmetic ------------------------------------------------------


def test_window_runs_from_reset_day_to_the_next():
    billing = BillingConfig(reset_day=15)
    start, end = billing.window(at(2026, 3, 20))
    assert day(start) == "2026-03-15"
    assert day(end) == "2026-04-15"


def test_window_rolls_back_before_the_reset_day_in_the_same_month():
    billing = BillingConfig(reset_day=15)
    start, end = billing.window(at(2026, 3, 10))
    assert day(start) == "2026-02-15"
    assert day(end) == "2026-03-15"


def test_window_rolls_over_the_year_boundary():
    billing = BillingConfig(reset_day=15)
    start, end = billing.window(at(2026, 1, 10))
    assert day(start) == "2025-12-15"
    assert day(end) == "2026-01-15"


def test_a_reset_day_past_the_month_end_is_clamped_not_rejected():
    """Billing on the 31st must mean the last day of February, not an error."""
    billing = BillingConfig(reset_day=31)
    start, end = billing.window(at(2026, 2, 10))
    assert day(start) == "2026-01-31"
    assert day(end) == "2026-02-28"


def test_explicit_period_overrides_the_reset_day():
    billing = BillingConfig(reset_day=1, period_start="2026-05-02", period_end="2026-05-09")
    start, end = billing.window(at(2026, 6, 20))
    assert day(start) == "2026-05-02"
    # Inclusive end: the budget covers the whole of the final day.
    assert datetime.fromtimestamp(end, UTC).date().isoformat() == "2026-05-09"


def test_no_cycle_configured_means_no_window():
    assert BillingConfig().window(at(2026, 3, 20)) is None


def test_is_current_is_false_for_a_stale_explicit_period():
    billing = BillingConfig(period_start="2026-01-01", period_end="2026-01-31")
    assert billing.is_current(at(2026, 1, 15))
    assert not billing.is_current(at(2026, 3, 15))


# -- metering gate ----------------------------------------------------------


def test_metering_needs_prices_and_a_cycle():
    cycle = BillingConfig(reset_day=1)
    assert SpendLedger(None, priced=True).metering is False  # no cycle at all
    assert SpendLedger(cycle, priced=False).metering is False  # no prices
    assert SpendLedger(cycle, priced=True).metering is True


def test_prices_without_a_reset_day_do_not_meter():
    """An explicit period is the other way to have a cycle; bare prices are not."""
    assert SpendLedger(BillingConfig(), priced=True).metering is False


# -- budget resolution ------------------------------------------------------


def test_budget_most_specific_key_wins():
    budgets = BillingConfig(
        reset_day=1,
        budgets=(
            CostEntry("*", "gpt-4o", 5.0, 0.0),
            CostEntry("openai", "gpt-4o", 50.0, 0.0),
        ),
    )
    table = BudgetTable(budgets.budgets)
    assert table.get("openai", "gpt-4o") == 50.0
    assert table.get("groq", "gpt-4o") == 5.0


def test_budget_whole_provider_key_matches_any_model():
    budgets = BillingConfig(reset_day=1, budgets=(CostEntry("openai", "*", 100.0, 0.0),))
    table = BudgetTable(budgets.budgets)
    assert table.get("openai", "anything") == 100.0
    assert table.get("groq", "anything") is None


# -- recording and balance --------------------------------------------------


def test_balance_is_cap_minus_spend():
    billing = BillingConfig(reset_day=1, budgets=(CostEntry("openai", "gpt-4o", 10.0, 0.0),))
    ledger = SpendLedger(billing, priced=True)
    ledger.record("openai", "gpt-4o", 2.5)
    ledger.record("openai", "gpt-4o", 1.5)
    assert ledger.spend("openai", "gpt-4o") == 4.0
    assert ledger.balance("openai", "gpt-4o") == 6.0


def test_an_undeclared_budget_has_no_balance():
    ledger = SpendLedger(BillingConfig(reset_day=1), priced=True)
    ledger.record("openai", "gpt-4o", 9.0)
    assert ledger.balance("openai", "gpt-4o") is None


def test_calls_are_counted_even_without_cost_data():
    """`least_used` ranks on calls and needs no price table at all."""
    ledger = SpendLedger(None, priced=False)
    ledger.record("openai", "gpt-4o")
    ledger.record("openai", "gpt-4o")
    assert ledger.calls("openai", "gpt-4o") == 2
    assert ledger.spend("openai", "gpt-4o") == 0.0


def test_a_recorded_call_without_a_model_is_ignored():
    ledger = SpendLedger(None)
    ledger.record("openai", None)
    ledger.record(None, "gpt-4o")
    assert ledger.calls("openai", "gpt-4o") == 0


def test_rollover_clears_last_cycles_spend():
    """A new window must not subtract last month's spend from this month's cap."""
    billing = BillingConfig(reset_day=15, budgets=(CostEntry("openai", "gpt-4o", 10.0, 0.0),))
    ledger = SpendLedger(billing, priced=True)
    ledger.record("openai", "gpt-4o", 4.0, at=at(2026, 2, 20))  # Feb 15 - Mar 15
    assert ledger.balance("openai", "gpt-4o", at=at(2026, 2, 25)) == 6.0
    # A read in the next window rolls the ledger and reports a full budget.
    assert ledger.spend("openai", "gpt-4o", at=at(2026, 3, 20)) == 0.0
    assert ledger.balance("openai", "gpt-4o", at=at(2026, 3, 20)) == 10.0


def test_describe_reports_the_window_and_totals():
    billing = BillingConfig(reset_day=15, budgets=(CostEntry("openai", "gpt-4o", 10.0, 0.0),))
    ledger = SpendLedger(billing, priced=True)
    ledger.record("openai", "gpt-4o", 2.0, at=at(2026, 2, 20))
    described = ledger.describe(at=at(2026, 2, 25))
    assert described["reset_day"] == 15
    assert described["total_spend"] == 2.0
    assert described["spend"] == {"openai/gpt-4o": 2.0}
    assert described["process_local"] is True
