"""Run budgets count real usage and never imply a spend cap."""

from __future__ import annotations

import threading

import pytest

from fx1.interactive.usage_budget import (
    PRESET_CHAT_TURN,
    PRESET_EXEC,
    PRESET_SUPERPOWER,
    BudgetExceeded,
    UsageLimits,
    UsageTracker,
)


class Mono:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_presets_are_operational_ceilings_not_prices() -> None:
    assert UsageLimits(180.0, 6, 4, 1, 24_000) == PRESET_CHAT_TURN
    assert UsageLimits(900.0, 16, 8, 2, 80_000) == PRESET_SUPERPOWER
    assert UsageLimits(300.0, 2, 1, 0, 4_000) == PRESET_EXEC
    for preset in (PRESET_CHAT_TURN, PRESET_SUPERPOWER, PRESET_EXEC):
        text = UsageTracker(preset).describe()
        assert "unit price is unknown" in text
        assert "no spend cap is implied" in text
        assert "not started" in text
        assert "Sharpe" not in text
        assert "P&L" not in text
        assert "$" not in text


def test_from_dict_roundtrip_and_missing_means_unconstrained() -> None:
    raw = {
        "max_wall_s": 1.5,
        "max_steps": 2,
        "max_model_calls": None,
        "max_retries_per_step": 0,
        "max_total_tokens": 10,
    }
    budget = UsageLimits.from_dict(raw)
    assert budget.to_dict() == raw
    assert UsageLimits.from_dict(budget.to_dict()) == budget
    assert UsageLimits.from_dict({}).to_dict() == UsageLimits().to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {"max_steps": True},
        {"max_steps": -1},
        {"max_steps": 1.5},
        {"max_wall_s": "10"},
        {"max_wall_s": float("inf")},
        {"max_wall_s": -0.1},
        {"max_total_tokens": False},
        {"nope": 1},
        ["max_steps"],
    ],
)
def test_from_dict_rejects_bad_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        UsageLimits.from_dict(payload)  # type: ignore[arg-type]


def test_type_must_be_a_budget() -> None:
    with pytest.raises(TypeError):
        UsageTracker({"max_steps": 1})  # type: ignore[arg-type]


def test_each_count_allows_the_ceiling_then_raises() -> None:
    tracker = UsageTracker(
        UsageLimits(max_steps=2, max_model_calls=1, max_total_tokens=5, max_retries_per_step=1)
    )
    tracker.charge_step(2)
    tracker.charge_model_call()
    tracker.charge_usage({"total_tokens": 5})
    tracker.charge_retry("vol-bench")
    tracker.check()
    with pytest.raises(BudgetExceeded) as steps:
        tracker.charge_step()
    assert steps.value.kind == "steps"
    assert steps.value.limit == 2
    assert steps.value.used == 3
    fresh = UsageTracker(UsageLimits(max_model_calls=1))
    fresh.charge_model_call()
    with pytest.raises(BudgetExceeded) as calls:
        fresh.charge_model_call()
    assert calls.value.kind == "model_calls" and calls.value.used == 2
    tokens = UsageTracker(UsageLimits(max_total_tokens=5))
    tokens.charge_usage({"total_tokens": 5})
    with pytest.raises(BudgetExceeded) as over:
        tokens.charge_usage({"prompt_tokens": 1, "completion_tokens": 1})
    assert over.value.kind == "tokens" and over.value.used == 7
    retries = UsageTracker(UsageLimits(max_retries_per_step=1))
    retries.charge_retry("a")
    retries.charge_retry("b")
    with pytest.raises(BudgetExceeded) as again:
        retries.charge_retry("a")
    assert again.value.kind == "retries" and again.value.limit == 1 and again.value.used == 2


def test_zero_ceilings_fail_on_the_first_charge_or_start() -> None:
    wall = UsageTracker(UsageLimits(max_wall_s=0), monotonic=Mono(0))
    wall.check()
    wall.start()
    with pytest.raises(BudgetExceeded) as info:
        wall.check()
    assert info.value.kind == "wall" and info.value.used == 0 and info.value.limit == 0
    steps = UsageTracker(UsageLimits(max_steps=0))
    with pytest.raises(BudgetExceeded) as stepped:
        steps.charge_step()
    assert stepped.value.kind == "steps" and stepped.value.used == 1
    retries = UsageTracker(UsageLimits(max_retries_per_step=0))
    with pytest.raises(BudgetExceeded) as retried:
        retries.charge_retry("only")
    assert retried.value.kind == "retries"


def test_wall_deadline_uses_the_injected_monotonic_clock() -> None:
    mono = Mono(10.0)
    tracker = UsageTracker(UsageLimits(max_wall_s=5), monotonic=mono)
    assert tracker.remaining_s() == 5
    assert tracker.deadline() is None
    tracker.check()
    tracker.start()
    assert tracker.deadline() == 15
    mono.now = 12
    tracker.start()
    assert tracker.deadline() == 15
    assert tracker.remaining_s() == pytest.approx(3)
    mono.now = 14.9
    tracker.check()
    assert tracker.remaining_s() == pytest.approx(0.1)
    mono.now = 15
    assert tracker.remaining_s() == 0
    with pytest.raises(BudgetExceeded) as info:
        tracker.check()
    assert info.value.kind == "wall"
    assert info.value.limit == 5
    assert info.value.used == 5
    assert isinstance(info.value, RuntimeError)


def test_unconstrained_budget_does_not_invent_a_deadline() -> None:
    mono = Mono(3)
    tracker = UsageTracker(UsageLimits(), monotonic=mono)
    tracker.start()
    mono.now = 100
    tracker.charge_step(4)
    tracker.charge_model_call()
    tracker.charge_retry("s")
    tracker.charge_usage({"total_tokens": 9})
    tracker.check()
    assert tracker.remaining_s() is None
    assert tracker.deadline() is None
    assert tracker.request_max_tokens(None) is None
    assert tracker.request_max_tokens(12) == 12


def test_request_max_tokens_caps_by_remaining_and_does_not_mutate() -> None:
    tracker = UsageTracker(UsageLimits(max_total_tokens=100))
    assert tracker.request_max_tokens(None) == 100
    assert tracker.request_max_tokens(40) == 40
    assert tracker.request_max_tokens(150) == 100
    assert tracker.request_max_tokens(0) == 0
    assert tracker.snapshot()["total_tokens"] == 0
    tracker.charge_usage({"total_tokens": 80})
    assert tracker.request_max_tokens(50) == 20
    assert tracker.request_max_tokens() == 20
    tracker.charge_usage({"total_tokens": 20})
    with pytest.raises(BudgetExceeded) as info:
        tracker.request_max_tokens(5)
    assert info.value.kind == "tokens" and info.value.used == 100
    with pytest.raises(ValueError):
        tracker.request_max_tokens(-1)
    with pytest.raises(ValueError):
        tracker.request_max_tokens(True)  # type: ignore[arg-type]
    assert tracker.snapshot()["total_tokens"] == 100


def test_usage_accounting_never_fabricates_omitted_totals() -> None:
    tracker = UsageTracker(UsageLimits(max_total_tokens=100))
    tracker.charge_model_call()
    tracker.charge_usage(None)
    tracker.charge_usage({})
    assert tracker.snapshot()["model_calls"] == 1
    assert tracker.snapshot()["total_tokens"] == 0
    assert tracker.snapshot()["unmetered_calls"] == 2
    tracker.charge_usage({"total_tokens": 0})
    tracker.charge_usage({"prompt_tokens": 2, "completion_tokens": 3})
    tracker.charge_usage({"prompt_tokens": 3})
    tracker.charge_usage({"total_tokens": 4, "prompt_tokens": 100, "completion_tokens": 100})
    snap = tracker.snapshot()
    assert snap["total_tokens"] == 12
    assert snap["unmetered_calls"] == 2
    assert snap["model_calls"] == 1
    text = tracker.describe()
    assert "tokens 12/100" in text
    assert "2 calls reported no usage" in text
    assert "unit price is unknown so no spend cap is implied" in text
    with pytest.raises(ValueError):
        tracker.charge_usage({"total_tokens": -1})
    with pytest.raises(ValueError):
        tracker.charge_usage({"total_tokens": True})  # type: ignore[dict-item]
    assert tracker.snapshot()["total_tokens"] == 12
    with pytest.raises(ValueError):
        tracker.charge_usage(["nope"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tracker.charge_step(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tracker.charge_step(0)
    with pytest.raises(ValueError):
        tracker.charge_retry("  ")


def test_snapshot_restore_rebases_elapsed_wall_time() -> None:
    mono = Mono(100.0)
    tracker = UsageTracker(
        UsageLimits(max_wall_s=10, max_total_tokens=50, max_steps=4), monotonic=mono
    )
    tracker.start()
    tracker.charge_step()
    tracker.charge_usage({"total_tokens": 15})
    tracker.charge_usage(None)
    tracker.charge_retry("synthesize")
    mono.now = 104
    snap = tracker.snapshot()
    assert snap["version"] == 1
    assert snap["elapsed_s"] == 4
    assert snap["started"] is True
    restored = UsageTracker.restore(snap, monotonic=Mono(0.0))
    assert restored.budget == tracker.budget
    assert restored.remaining_s() == pytest.approx(6)
    assert restored.deadline() == pytest.approx(6)
    again = restored.snapshot()
    assert again["steps"] == 1
    assert again["total_tokens"] == 15
    assert again["unmetered_calls"] == 1
    assert again["retries"] == {"synthesize": 1}
    assert "1 calls reported no usage" in restored.describe()
    over = dict(snap)
    over["steps"] = 9
    held = UsageTracker.restore(over, monotonic=Mono(0.0))
    with pytest.raises(BudgetExceeded) as info:
        held.check()
    assert info.value.kind == "steps" and info.value.used == 9


def test_restore_rejects_bad_snapshots() -> None:
    tracker = UsageTracker(UsageLimits(max_steps=1), monotonic=Mono(0))
    tracker.start()
    snap = tracker.snapshot()
    with pytest.raises(ValueError):
        UsageTracker.restore({"version": 2})
    broken = dict(snap)
    broken["version"] = 2
    with pytest.raises(ValueError):
        UsageTracker.restore(broken)
    broken = dict(snap)
    del broken["steps"]
    with pytest.raises(ValueError):
        UsageTracker.restore(broken)
    broken = dict(snap)
    broken["started"] = "yes"
    with pytest.raises(ValueError):
        UsageTracker.restore(broken)
    broken = dict(snap)
    broken["elapsed_s"] = -1
    with pytest.raises(ValueError):
        UsageTracker.restore(broken)
    broken = dict(snap)
    broken["retries"] = {"ok": True}
    with pytest.raises(ValueError):
        UsageTracker.restore(broken)


def test_not_started_snapshot_restores_without_a_deadline() -> None:
    tracker = UsageTracker(UsageLimits(max_wall_s=8), monotonic=Mono(50))
    restored = UsageTracker.restore(tracker.snapshot(), monotonic=Mono(1))
    assert restored.snapshot()["started"] is False
    assert restored.snapshot()["elapsed_s"] == 0
    assert restored.deadline() is None
    assert restored.remaining_s() == 8
    restored.check()


def test_concurrent_charges_sum_exactly() -> None:
    tracker = UsageTracker(
        UsageLimits(
            max_steps=10_000,
            max_model_calls=10_000,
            max_total_tokens=10_000_000,
            max_retries_per_step=10_000,
        )
    )
    threads_n, per_thread = 4, 30

    def work() -> None:
        for _ in range(per_thread):
            tracker.charge_step()
            tracker.charge_model_call()
            tracker.charge_usage({"total_tokens": 2})
            tracker.charge_retry("s")

    threads = [threading.Thread(target=work) for _ in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    snap = tracker.snapshot()
    assert snap["steps"] == threads_n * per_thread
    assert snap["model_calls"] == threads_n * per_thread
    assert snap["total_tokens"] == threads_n * per_thread * 2
    assert snap["retries"] == {"s": threads_n * per_thread}
    assert snap["unmetered_calls"] == 0


def test_a_late_charge_still_reports_the_dimension_already_over() -> None:
    tracker = UsageTracker(UsageLimits(max_retries_per_step=1, max_steps=10))
    tracker.charge_retry("a")
    with pytest.raises(BudgetExceeded):
        tracker.charge_retry("a")
    with pytest.raises(BudgetExceeded) as info:
        tracker.charge_step()
    assert info.value.kind == "retries"
    assert tracker.snapshot()["steps"] == 1
