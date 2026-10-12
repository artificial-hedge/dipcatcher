"""Run budgets: validation, environment overrides, and fail-closed ledger charges."""

from __future__ import annotations

import threading

import pytest

from fx1.interactive.budget import BudgetExceeded, BudgetLedger, RunBudget

_ENV_VARS = (
    "FX1_BUDGET_WALL_S",
    "FX1_BUDGET_HARNESS_STEPS",
    "FX1_BUDGET_MODEL_CALLS",
    "FX1_BUDGET_RETRIES",
    "FX1_BUDGET_WEB_SOURCES",
    "FX1_BUDGET_PROMPT_TOKENS",
)


def test_validate_accepts_defaults_and_edges() -> None:
    RunBudget().validate()
    RunBudget(max_retries_per_call=0, max_prompt_tokens=None).validate()
    RunBudget(max_prompt_tokens=1, max_wall_s=0.1).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_wall_s": 0},
        {"max_wall_s": -1},
        {"max_wall_s": float("nan")},
        {"max_wall_s": float("inf")},
        {"max_harness_steps": 0},
        {"max_harness_steps": -2},
        {"max_model_calls": 0},
        {"max_retries_per_call": -1},
        {"max_web_sources": 0},
        {"max_prompt_tokens": 0},
        {"max_prompt_tokens": -4},
    ],
)
def test_validate_rejects_each_bad_field(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RunBudget(**kwargs).validate()  # type: ignore[arg-type]


def test_describe_default_and_capped_tokens() -> None:
    assert RunBudget().describe() == (
        "wall 1800s · harness steps 4 · model calls 8 · retries/call 2 · "
        "web sources 8 · prompt tokens: uncapped (provider limits unknown)"
    )
    capped = RunBudget(
        max_wall_s=12.5,
        max_retries_per_call=0,
        max_prompt_tokens=4096,
    )
    assert capped.describe() == (
        "wall 12.5s · harness steps 4 · model calls 8 · retries/call 0 · "
        "web sources 8 · prompt tokens: 4096"
    )


def test_from_env_defaults_and_overrides() -> None:
    assert RunBudget.from_env({}) == RunBudget()
    budget = RunBudget.from_env(
        {
            "FX1_BUDGET_WALL_S": "12.5",
            "FX1_BUDGET_HARNESS_STEPS": "2",
            "FX1_BUDGET_MODEL_CALLS": "3",
            "FX1_BUDGET_RETRIES": "0",
            "FX1_BUDGET_WEB_SOURCES": "4",
            "FX1_BUDGET_PROMPT_TOKENS": "500",
            "UNRELATED": "ignored",
        }
    )
    assert budget == RunBudget(
        max_wall_s=12.5,
        max_harness_steps=2,
        max_model_calls=3,
        max_retries_per_call=0,
        max_web_sources=4,
        max_prompt_tokens=500,
    )


def test_from_env_reads_process_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FX1_BUDGET_MODEL_CALLS", "6")
    budget = RunBudget.from_env()
    assert budget.max_model_calls == 6
    assert budget.max_wall_s == 1800.0
    assert budget.max_harness_steps == 4
    assert budget.max_prompt_tokens is None


@pytest.mark.parametrize(
    ("var", "raw"),
    [
        ("FX1_BUDGET_WALL_S", "nope"),
        ("FX1_BUDGET_WALL_S", "nan"),
        ("FX1_BUDGET_WALL_S", "0"),
        ("FX1_BUDGET_WALL_S", "-5"),
        ("FX1_BUDGET_HARNESS_STEPS", "x"),
        ("FX1_BUDGET_HARNESS_STEPS", "0"),
        ("FX1_BUDGET_MODEL_CALLS", ""),
        ("FX1_BUDGET_RETRIES", "-1"),
        ("FX1_BUDGET_WEB_SOURCES", "1.5"),
        ("FX1_BUDGET_PROMPT_TOKENS", "0"),
        ("FX1_BUDGET_PROMPT_TOKENS", "lots"),
    ],
)
def test_from_env_invalid_names_the_variable(var: str, raw: str) -> None:
    with pytest.raises(ValueError, match=var):
        RunBudget.from_env({var: raw})


def test_budget_exceeded_message() -> None:
    err = BudgetExceeded("harness_steps", 4, 5)
    assert isinstance(err, RuntimeError)
    assert err.dimension == "harness_steps"
    assert err.limit == 4
    assert err.used == 5
    assert str(err) == "budget exceeded: harness_steps 5 > 4"


def test_ledger_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError):
        BudgetLedger(RunBudget(max_web_sources=0))


def test_wall_deadline_with_fake_clock() -> None:
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    ledger = BudgetLedger(RunBudget(max_wall_s=10), clock=clock)
    assert ledger.budget.max_wall_s == 10
    assert ledger.elapsed_s() == 0.0
    assert ledger.remaining_s() == 10.0
    assert ledger.deadline_reached() is False
    ledger.check_wall()
    now["t"] = 10.0
    assert ledger.deadline_reached() is True
    assert ledger.remaining_s() == 0.0
    with pytest.raises(BudgetExceeded) as caught:
        ledger.check_wall()
    assert caught.value.dimension == "wall_s"
    assert caught.value.limit == 10
    assert caught.value.used == 10.0
    now["t"] = 12.5
    assert ledger.elapsed_s() == 12.5
    assert ledger.remaining_s() == 0.0
    with pytest.raises(BudgetExceeded) as later:
        ledger.check_wall()
    assert later.value.used == 12.5
    assert str(later.value) == "budget exceeded: wall_s 12.5 > 10"


def test_harness_step_and_web_source_boundaries() -> None:
    ledger = BudgetLedger(RunBudget(max_harness_steps=2, max_web_sources=1))
    ledger.charge_harness_step("a")
    ledger.charge_harness_step("a")
    with pytest.raises(BudgetExceeded) as steps:
        ledger.charge_harness_step("b")
    assert steps.value.dimension == "harness_steps"
    assert steps.value.limit == 2
    assert steps.value.used == 3
    assert str(steps.value) == "budget exceeded: harness_steps 3 > 2"
    assert ledger.snapshot()["harness_steps"]["ids"] == ["a", "a"]
    ids = ledger.snapshot()["harness_steps"]["ids"]
    assert isinstance(ids, list)
    ids.append("nope")
    assert ledger.snapshot()["harness_steps"]["ids"] == ["a", "a"]

    ledger.charge_web_source()
    with pytest.raises(BudgetExceeded) as web:
        ledger.charge_web_source()
    assert web.value.dimension == "web_sources"
    assert web.value.limit == 1
    assert web.value.used == 2
    assert ledger.snapshot()["web_sources"] == {"used": 1, "limit": 1}


def test_retry_boundary_is_per_call() -> None:
    ledger = BudgetLedger(RunBudget(max_retries_per_call=2))
    ledger.charge_retry("c1")
    ledger.charge_retry("c1")
    ledger.charge_retry("c2")
    with pytest.raises(BudgetExceeded) as caught:
        ledger.charge_retry("c1")
    assert caught.value.dimension == "retries"
    assert caught.value.limit == 2
    assert caught.value.used == 3
    assert ledger.snapshot()["retries"] == {"c1": 2, "c2": 1}

    blocked = BudgetLedger(RunBudget(max_retries_per_call=0))
    with pytest.raises(BudgetExceeded) as zero:
        blocked.charge_retry("c")
    assert zero.value.used == 1
    assert zero.value.limit == 0
    assert blocked.snapshot()["retries"] == {}


def test_model_call_boundary() -> None:
    ledger = BudgetLedger(RunBudget(max_model_calls=2, max_prompt_tokens=None))
    ledger.charge_model_call(prompt_tokens=10**9, completion_tokens=3)
    ledger.charge_model_call(prompt_tokens=1, completion_tokens=4)
    with pytest.raises(BudgetExceeded) as caught:
        ledger.charge_model_call(prompt_tokens=5, completion_tokens=9)
    assert caught.value.dimension == "model_calls"
    assert caught.value.limit == 2
    assert caught.value.used == 3
    snap = ledger.snapshot()
    assert snap["model_calls"] == {"used": 2, "limit": 2}
    assert snap["tokens"]["prompt"] == 10**9 + 1
    assert snap["tokens"]["completion"] == 7
    assert snap["tokens"]["prompt_limit"] is None


def test_prompt_token_cap_only_when_set() -> None:
    open_ledger = BudgetLedger(RunBudget(max_prompt_tokens=None, max_model_calls=4))
    open_ledger.charge_model_call(prompt_tokens=10_000_000)
    assert open_ledger.snapshot()["tokens"]["prompt"] == 10_000_000

    ledger = BudgetLedger(RunBudget(max_prompt_tokens=100, max_model_calls=5))
    ledger.charge_model_call(prompt_tokens=100, completion_tokens=7)
    ledger.charge_model_call(prompt_tokens=0, completion_tokens=3)
    with pytest.raises(BudgetExceeded) as caught:
        ledger.charge_model_call(prompt_tokens=1, completion_tokens=99)
    assert caught.value.dimension == "prompt_tokens"
    assert caught.value.limit == 100
    assert caught.value.used == 101
    snap = ledger.snapshot()
    assert snap["model_calls"]["used"] == 2
    assert snap["tokens"] == {"prompt": 100, "completion": 10, "prompt_limit": 100}


def test_negative_token_counts_rejected() -> None:
    ledger = BudgetLedger(RunBudget())
    with pytest.raises(ValueError, match="prompt_tokens"):
        ledger.charge_model_call(prompt_tokens=-1)
    with pytest.raises(ValueError, match="completion_tokens"):
        ledger.charge_model_call(completion_tokens=-1)
    assert ledger.snapshot()["model_calls"]["used"] == 0


def test_snapshot_and_describe() -> None:
    now = {"t": 5.0}

    def clock() -> float:
        return now["t"]

    ledger = BudgetLedger(RunBudget(max_wall_s=20, max_prompt_tokens=50), clock=clock)
    now["t"] = 8.0
    ledger.charge_harness_step("s1")
    ledger.charge_model_call(prompt_tokens=10, completion_tokens=4)
    ledger.charge_retry("call-a")
    ledger.charge_web_source()
    assert ledger.snapshot() == {
        "elapsed_s": 3.0,
        "remaining_s": 17.0,
        "harness_steps": {"used": 1, "limit": 4, "ids": ["s1"]},
        "model_calls": {"used": 1, "limit": 8},
        "tokens": {"prompt": 10, "completion": 4, "prompt_limit": 50},
        "retries": {"call-a": 1},
        "web_sources": {"used": 1, "limit": 8},
    }

    started = {"t": 0.0}

    def moving() -> float:
        return started["t"]

    described = BudgetLedger(RunBudget(), clock=moving)
    started["t"] = 12.0
    described.charge_harness_step("step")
    described.charge_model_call(prompt_tokens=1234, completion_tokens=56)
    described.charge_model_call()
    described.charge_web_source()
    described.charge_web_source()
    described.charge_web_source()
    assert described.describe() == (
        "budget: 12s/1800s wall · 1/4 harness · 2/8 model calls · tokens p=1234 c=56 · web 3/8"
    )


def test_model_call_charges_are_thread_safe() -> None:
    limit = 8
    attempts = 32
    ledger = BudgetLedger(RunBudget(max_model_calls=limit))
    barrier = threading.Barrier(8)
    successes: list[int] = []
    failures: list[BudgetExceeded] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait(timeout=5)
        for _ in range(attempts):
            try:
                ledger.charge_model_call(prompt_tokens=1, completion_tokens=1)
            except BudgetExceeded as exc:
                with guard:
                    failures.append(exc)
            else:
                with guard:
                    successes.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert thread.is_alive() is False
    assert len(successes) == limit
    assert len(failures) == 8 * attempts - limit
    assert ledger.snapshot()["model_calls"]["used"] == limit
    assert ledger.snapshot()["tokens"]["prompt"] == limit
    assert all(exc.dimension == "model_calls" for exc in failures)
