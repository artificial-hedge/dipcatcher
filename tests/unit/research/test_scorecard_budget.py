"""CI scorecard budgets shrink pytest fixtures and leave production defaults."""

from __future__ import annotations

from quant_fund.research import scorecard_budget
from quant_fund.research.scorecard_budget import cached_optional_blobs, call_bench


def bench_decision_transformer(seed: int, n_ep: int = 200, iters: int = 1600) -> dict[str, int]:
    return {"seed": seed, "n_ep": n_ep, "iters": iters}


def test_full_budget_keeps_production_defaults(monkeypatch) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "full")
    out = call_bench(bench_decision_transformer, seed=3)
    assert out == {"seed": 3, "n_ep": 200, "iters": 1600}


def test_ci_budget_shrinks_the_scorecard_fixture(monkeypatch) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "ci")
    out = call_bench(bench_decision_transformer, seed=3)
    assert out == {"seed": 3, "n_ep": 30, "iters": 60}


def test_ci_budget_reuses_the_optional_blob(monkeypatch) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "ci")
    scorecard_budget._CI_OPTIONAL_BLOBS.clear()
    calls = {"n": 0}

    def build() -> dict[str, dict[str, int]]:
        calls["n"] += 1
        return {"family": {"score": calls["n"]}}

    first = cached_optional_blobs(build)
    first["family"]["score"] = 99
    second = cached_optional_blobs(build)
    assert calls["n"] == 1
    assert second == {"family": {"score": 1}}
    assert first["family"]["score"] == 99


def test_full_budget_does_not_cache_the_optional_blob(monkeypatch) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "full")
    scorecard_budget._CI_OPTIONAL_BLOBS.clear()
    calls = {"n": 0}

    def build() -> dict[str, int]:
        calls["n"] += 1
        return {"n": calls["n"]}

    assert cached_optional_blobs(build) == {"n": 1}
    assert cached_optional_blobs(build) == {"n": 2}


def test_explicit_kwargs_win_over_the_ci_map(monkeypatch) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "ci")
    out = call_bench(bench_decision_transformer, seed=3, iters=5)
    assert out == {"seed": 3, "n_ep": 30, "iters": 5}
