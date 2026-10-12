"""Hard ceilings for /superpower planning through synthesis. Offline; no network."""

from __future__ import annotations

import pytest

from fx1.interactive.budgets import BudgetError, RunBudget, merge_budget

_NO_DOLLAR_CAP = (
    "no price catalog is configured; this budget is time/step limits only, not a dollar cap"
)


def test_validate_rejects_negative_ceilings() -> None:
    for name in ("max_steps", "max_seconds", "max_retries", "max_families", "max_harness"):
        budget = RunBudget()
        setattr(budget, name, -1)
        with pytest.raises(BudgetError, match="must not be negative"):
            budget.validate()


def test_validate_rejects_count_and_time_above_hard_max() -> None:
    for name in ("max_steps", "max_families", "max_harness"):
        budget = RunBudget()
        setattr(budget, name, 33)
        with pytest.raises(BudgetError, match="exceeds 32"):
            budget.validate()
    budget = RunBudget()
    budget.max_seconds = 86401
    with pytest.raises(BudgetError, match="exceeds 86400"):
        budget.validate()
    RunBudget(max_steps=32, max_families=32, max_harness=32, max_seconds=86400).validate()


def test_charge_step_stops_at_max_steps() -> None:
    budget = RunBudget(max_steps=2)
    budget.charge_step()
    budget.charge_step()
    assert budget.steps == 2
    with pytest.raises(BudgetError, match="step budget exceeded"):
        budget.charge_step()
    assert budget.steps == 2


def test_expired_at_time_ceiling_blocks_another_step(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = RunBudget(max_seconds=10, started_at=1_000.0, max_steps=4)
    assert budget.expired(now=1_009.999) is False
    assert budget.expired(now=1_010.0) is True
    monkeypatch.setattr("fx1.interactive.budgets.time.monotonic", lambda: 1_010.0)
    with pytest.raises(BudgetError, match="expired"):
        budget.charge_step()
    assert budget.steps == 0


def test_charge_retry_stops_at_max_retries() -> None:
    budget = RunBudget(max_retries=1)
    budget.charge_retry()
    assert budget.retries == 1
    with pytest.raises(BudgetError, match="retry budget exceeded"):
        budget.charge_retry()
    assert budget.retries == 1


def test_as_public_spend_cap_is_none_and_denies_a_dollar_cap() -> None:
    budget = RunBudget()
    budget.charge_step()
    public = budget.as_public()
    assert public["spend_cap"] is None
    assert public["spend_cap_note"] == _NO_DOLLAR_CAP
    assert "not a dollar cap" in public["spend_cap_note"]
    assert public["steps"] == 1
    assert public["retries"] == 0
    assert public["expired"] is False
    assert public["max_steps"] == 12
    assert public["max_seconds"] == 3600
    assert public["max_retries"] == 1
    assert public["max_families"] == 4
    assert public["max_harness"] == 4
    assert "started_at" not in public


def test_merge_rejects_unknown_override_key() -> None:
    base = RunBudget()
    base.charge_step()
    with pytest.raises(BudgetError, match="unknown budget override: spend_cap"):
        merge_budget(base, {"spend_cap": 25})
    with pytest.raises(BudgetError, match="price"):
        merge_budget(base, {"max_steps": 4, "price": 10})
    assert base.steps == 1
    assert base.max_steps == 12


def test_merge_applies_allowed_overrides_and_resets_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 500.0}

    def monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr("fx1.interactive.budgets.time.monotonic", monotonic)
    base = RunBudget(max_families=3)
    base.charge_step()
    base.charge_retry()
    merged = merge_budget(base, {"max_steps": 6, "max_harness": 2})
    assert merged.steps == 0
    assert merged.retries == 0
    assert merged.max_steps == 6
    assert merged.max_harness == 2
    assert merged.max_families == 3
    assert merged.started_at > base.started_at
    assert base.steps == 1
    assert base.retries == 1
    assert merged.as_public()["spend_cap"] is None
    with pytest.raises(BudgetError):
        merge_budget(base, {"max_retries": -1})
