"""SYNTHETIC numerical/optimization correctness, never market or device evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.portfolio.qubo_selection import (
    AnnealConfig,
    CovarianceEvidence,
    ForecastEvidence,
    QUBOModel,
    SelectionProblem,
    exact_selection,
    greedy_selection,
    restore_receipt,
    save_receipt,
    solve_selection,
)

NOW = datetime(2025, 2, 1, 12, tzinfo=UTC)
SHA = hashlib.sha256(b"SYNTHETIC QUBO problem generator").hexdigest()


def problem(n: int = 5, k: int = 2) -> SelectionProblem:
    assets = tuple(f"SYNTHETIC_{i}" for i in range(n))
    covariance = np.eye(n) * 0.02 + np.ones((n, n)) * 0.002
    return SelectionProblem(
        ForecastEvidence(
            assets,
            tuple(0.03 + 0.05 * math.sin(i) for i in range(n)),
            timedelta(days=1),
            NOW - timedelta(minutes=2),
            NOW - timedelta(minutes=1),
            "SYNTHETIC_FORECAST",
            SHA,
            True,
        ),
        CovarianceEvidence(
            assets,
            tuple(tuple(float(v) for v in row) for row in covariance),
            timedelta(days=1),
            NOW - timedelta(hours=1),
            NOW - timedelta(minutes=1),
            "SYNTHETIC_COVARIANCE",
            SHA,
            True,
        ),
        NOW,
        k,
        risk_aversion=2.0,
        budget=0.8,
    )


def small_config(**kwargs: Any) -> AnnealConfig:
    return AnnealConfig(seeds=(3, 19), sweeps=15, **kwargs)


def objective(p: SelectionProblem, bits: tuple[int, ...]) -> float:
    # Independent original mean/variance expression; no production QUBO helpers.
    weights = p.budget / p.cardinality * np.array(bits)
    return float(
        p.risk_aversion * weights @ np.array(p.covariance.values) @ weights
        - np.array(p.forecast.expected_returns) @ weights
    )


def rehash(payload: dict[str, Any]) -> None:
    # Deliberately rehash a forgery: replay must reject even valid content hashes.
    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
        ).hexdigest()

    payload["receipt_sha256"] = digest(payload["body"])
    stable = json.loads(json.dumps(payload["body"]))
    for trial in stable["trials"]:
        trial.pop("elapsed_seconds")
    for name in ("greedy", "exact"):
        stable[name].pop("elapsed_seconds")
    for name in ("anneal_elapsed_seconds", "greedy_elapsed_seconds", "exact_elapsed_seconds"):
        stable["diagnostics"].pop(name)
    payload["deterministic_sha256"] = digest(stable)


def test_direct_objective_qubo_expansion_and_safe_penalty_on_entire_hypercube() -> None:
    p = problem()
    model = QUBOModel(p)
    feasible_energy, infeasible_energy = [], []
    for bits in itertools.product((0, 1), repeat=len(p.assets)):
        expected = objective(p, bits)
        assert p.objective(bits, require_feasible=False) == pytest.approx(expected)
        expected_energy = expected + float(model.penalty) * (sum(bits) - p.cardinality) ** 2
        assert model.energy(bits) == pytest.approx(expected_energy)
        assert np.array(bits) @ model.array() @ np.array(bits) + model.offset == pytest.approx(
            expected_energy, abs=1e-11
        )
        assert abs(expected) <= model.objective_absolute_bound + 1e-12
        (feasible_energy if sum(bits) == p.cardinality else infeasible_energy).append(
            model.energy(bits)
        )
    assert min(infeasible_energy) > max(feasible_energy)
    with pytest.raises(ValueError, match="2B"):
        QUBOModel(p, penalty=2 * model.objective_absolute_bound)
    zero = replace(
        p,
        forecast=replace(p.forecast, expected_returns=(0.0,) * len(p.assets)),
        covariance=replace(p.covariance, values=tuple((0.0,) * len(p.assets) for _ in p.assets)),
    )
    assert QUBOModel(zero).penalty == 1.0


def test_exhaustive_crosscheck_is_independent_and_greedy_constraints_match() -> None:
    p = problem(n=7, k=3)
    candidates = [
        (objective(p, bits), bits) for bits in itertools.product((0, 1), repeat=7) if sum(bits) == 3
    ]
    exact = exact_selection(p)
    assert exact.objective == pytest.approx(min(value for value, _ in candidates))
    assert exact.evaluated_candidates == math.comb(7, 3) == len(candidates)
    assert exact.optimality_proven and exact.status == "EXHAUSTIVE_FLOAT64"
    greedy = greedy_selection(p)
    assert greedy.bits is not None and sum(greedy.bits) == 3
    assert greedy.evaluated_candidates == 7 + 6 + 5
    assert not greedy.optimality_proven
    assert greedy.objective == pytest.approx(objective(p, greedy.bits))
    assert float(greedy.objective) >= float(exact.objective) - 1e-12


@pytest.mark.parametrize(
    ("n", "k", "max_combinations", "status"),
    [(21, 1, 100_000, "SKIPPED_ASSET_LIMIT"), (8, 4, 10, "SKIPPED_COMBINATION_LIMIT")],
)
def test_exact_limits_never_fabricate_optimum(
    n: int, k: int, max_combinations: int, status: str
) -> None:
    result = exact_selection(problem(n, k), max_combinations=max_combinations)
    assert result.status == status and not result.optimality_proven
    assert result.bits is None and result.objective is None and result.evaluated_candidates == 0
    assert result.feasible_candidate_count == math.comb(n, k)


def test_real_metropolis_accepts_uphill_reproducibly_without_global_rng_changes() -> None:
    p = problem()
    config = replace(small_config(), initial_temperature=1000.0, final_temperature=1000.0)
    random_state = np.random.get_state()
    first = solve_selection(p, config)
    second = solve_selection(p, config)
    after = np.random.get_state()
    assert all(
        np.array_equal(before, current) for before, current in zip(random_state, after, strict=True)
    )
    assert first.deterministic_sha256 == second.deterministic_sha256
    assert len(first.receipt_sha256) == len(second.receipt_sha256) == 64
    assert first.diagnostics()["accepted_uphill"] > 0
    assert first.diagnostics()["proposal_count"] == 2 * 15 * len(p.assets)
    for trial in first.trials:
        assert len(trial.sweep_energy) == config.sweeps + 1
        assert trial.elapsed_seconds >= 0
        if trial.best_feasible_bits is not None:
            assert sum(trial.best_feasible_bits) == p.cardinality
    assert first.payload()["method"] == "classical_unconstrained_qubo_simulated_annealing"


def test_one_sweep_matches_independent_metropolis_transitions() -> None:
    p = problem(n=3, k=1)
    seed = 13
    config = AnnealConfig(seeds=(seed,), sweeps=1, initial_temperature=2.0, final_temperature=0.01)
    initial = (1, 0, 0)
    model = QUBOModel(p)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    expected = initial
    expected_energy = objective(p, expected)
    accepted = uphill = 0
    for _ in range(3):
        index = int(rng.integers(3))
        candidate = tuple(1 - v if i == index else v for i, v in enumerate(expected))
        energy = objective(p, candidate) + float(model.penalty) * (sum(candidate) - 1) ** 2
        delta = energy - expected_energy
        if delta <= 0 or float(rng.random()) < math.exp(-delta / 2.0):
            expected, expected_energy = candidate, energy
            accepted += 1
            uphill += int(delta > 0)
    actual = solve_selection(p, config, initial_states=(initial,)).trials[0]
    assert actual.final_bits == expected
    assert actual.final_energy == pytest.approx(expected_energy)
    assert (actual.accepted, actual.accepted_uphill) == (accepted, uphill)


def test_poor_anneal_outcome_and_objective_gap_are_retained() -> None:
    p = problem(n=4, k=1)
    p = replace(
        p,
        forecast=replace(p.forecast, expected_returns=(0.0, 0.9, 0.4, 0.2)),
        covariance=replace(p.covariance, values=((0.0,) * 4,) * 4),
        budget=1.0,
    )
    config = AnnealConfig(seeds=(1,), sweeps=2, initial_temperature=1e-12, final_temperature=1e-12)
    report = solve_selection(p, config, initial_states=((1, 0, 0, 0),))
    assert report.chosen_bits == (1, 0, 0, 0)
    assert report.exact.bits == (0, 1, 0, 0)
    assert report.trials[0].accepted == 0
    assert report.diagnostics()["anneal_objective"] == 0.0
    assert report.diagnostics()["anneal_objective_gap"] == pytest.approx(0.9)
    assert report.diagnostics()["constraints_matched"]
    assert not report.diagnostics()["compute_budget_matched"]
    assert not report.diagnostics()["objective_is_proper_score"]


def test_no_feasible_trial_is_not_replaced_with_baseline_success() -> None:
    p = problem(n=5, k=5)
    config = AnnealConfig(seeds=(0,), sweeps=1, initial_temperature=1e-12, final_temperature=1e-12)
    report = solve_selection(p, config, initial_states=((0, 0, 0, 0, 0),))
    assert report.chosen_bits is None and report.weights is None
    assert report.cash_weight is None
    assert report.trials[0].status == "NO_FEASIBLE_FOUND"
    assert report.trials[0].failure_reason is not None
    assert report.greedy.bits == report.exact.bits == (1, 1, 1, 1, 1)
    assert report.diagnostics()["failed_trials"] == 1
    assert report.diagnostics()["anneal_objective_gap"] is None


def test_swap_variant_is_feasible_and_bad_trial_is_retained_without_repair() -> None:
    p = problem()
    config = small_config(mode="cardinality_swap")
    report = solve_selection(p, config, initial_states=((0, 0, 0, 0, 0), (1, 1, 0, 0, 0)))
    assert report.trials[0].status == "INVALID_SWAP_INITIAL_STATE"
    assert report.trials[0].proposals == 0 and report.trials[0].best_feasible_bits is None
    assert report.trials[1].status == "FEASIBLE_FOUND"
    assert sum(report.trials[1].final_bits) == p.cardinality
    assert report.trials[1].feasible_visits == report.trials[1].accepted + 1
    assert report.payload()["method"] == "classical_cardinality_constrained_swap_annealing"
    assert report.diagnostics()["failed_trials"] == 1
    assert report.weights is not None and sum(report.weights) == pytest.approx(p.budget)
    assert report.cash_weight == pytest.approx(1.0 - p.budget)
    assert all(w == 0 or w == pytest.approx(p.budget / p.cardinality) for w in report.weights)
    singleton = solve_selection(problem(1, 1), small_config(mode="cardinality_swap"))
    assert singleton.diagnostics()["proposal_count"] == 0
    assert singleton.chosen_bits == (1,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": (True,)},
        {"seeds": (1, 1)},
        {"seeds": ()},
        {"sweeps": 0},
        {"sweeps": 10001},
        {"max_proposals": 2_000_001},
        {"initial_temperature": math.inf},
        {"final_temperature": 0.0},
        {"initial_temperature": 0.01, "final_temperature": 1.0},
        {"mode": "quantum_device"},
        {"exact_max_assets": 21},
        {"exact_max_combinations": 250001},
    ],
)
def test_config_fails_closed(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AnnealConfig(**kwargs)


def test_resource_initial_state_and_penalty_gates() -> None:
    p = problem()
    with pytest.raises(ValueError, match="resource budget"):
        solve_selection(p, replace(small_config(), max_proposals=1))
    for states in (((0, 1), (0, 1)), ((0, 0, 0, 0, 0),), ((True,) * 5,) * 2):
        with pytest.raises(ValueError):
            solve_selection(p, small_config(), initial_states=states)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2B"):
        solve_selection(p, small_config(), penalty=0.0)
    with pytest.raises(ValueError, match="cardinality"):
        p.weights((0, 0, 0, 0, 0))


def test_numeric_pit_alignment_and_source_schema_fail_closed() -> None:
    p = problem()
    for returns in ((math.nan,) * 5, (-1.1,) * 5, (True,) * 5, (1.0,)):
        with pytest.raises(ValueError):
            replace(p.forecast, expected_returns=returns)
    for assets in (("A", "A", "B", "C", "D"), ("", "A", "B", "C", "D")):
        with pytest.raises(ValueError):
            replace(p.forecast, assets=assets)
    for kwargs in (
        {"available_time": NOW - timedelta(days=2)},
        {"asof": NOW.replace(tzinfo=None)},
        {"source_sha256": "missing"},
        {"synthetic": 1},
        {"horizon": timedelta(0)},
    ):
        with pytest.raises(ValueError):
            replace(p.forecast, **kwargs)
    with pytest.raises(ValueError, match="unpublished"):
        replace(p, forecast=replace(p.forecast, available_time=NOW + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="unpublished"):
        replace(p, covariance=replace(p.covariance, available_time=NOW + timedelta(seconds=1)))
    with pytest.raises(ValueError, match="horizon"):
        replace(p, covariance=replace(p.covariance, horizon=timedelta(days=2)))
    with pytest.raises(ValueError, match="ordering"):
        replace(p, covariance=replace(p.covariance, assets=tuple(reversed(p.assets))))
    for kwargs in (
        {"cardinality": True},
        {"cardinality": 6},
        {"budget": 0.0},
        {"risk_aversion": -1.0},
        {"risk_aversion": 1e-320},
        {"budget": 1e-320},
    ):
        with pytest.raises(ValueError):
            replace(p, **kwargs)
    offset = timezone(timedelta(hours=5, minutes=30))
    canonical = replace(p, decision_time=NOW.astimezone(offset))
    assert canonical.decision_time == NOW
    assert canonical.problem_sha256 == p.problem_sha256


@pytest.mark.parametrize(
    "values",
    [((1.0, 0.2), (0.3, 1.0)), ((1.0, 2.0), (2.0, 1.0)), ((math.inf, 0.0), (0.0, 1.0)), ((1.0,),)],
)
def test_covariance_shape_symmetry_psd_and_finiteness(
    values: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ValueError):
        replace(problem(2, 1).covariance, values=values)


def test_immutable_arrays_receipt_hashes_and_explicit_research_identity() -> None:
    p = problem()
    report = solve_selection(p, small_config())
    copied = report.model.array()
    copied[:] = 0.0
    assert report.model.array().sum() != 0.0
    with pytest.raises(FrozenInstanceError):
        report.chosen_bits = (0,) * len(p.assets)  # type: ignore[misc]
    payload = report.payload()
    payload["trials"][0]["final_bits"] = (0,) * len(p.assets)
    assert report.payload()["trials"][0]["final_bits"] != payload["trials"][0]["final_bits"]
    assert report.research_only and not report.market_evidence
    assert not report.quantum_device and not report.quantum_advantage
    assert report.payload()["data_label"] == "SYNTHETIC"
    changed = replace(p, forecast=replace(p.forecast, source_sha256="0" * 64))
    assert changed.problem_sha256 != p.problem_sha256
    assert QUBOModel(changed).model_sha256 != report.model.model_sha256
    mixed = replace(p, forecast=replace(p.forecast, synthetic=False))
    assert solve_selection(mixed, small_config()).payload()["data_label"] == "MIXED"


def test_write_once_json_receipt_restores_by_replaying_solver(tmp_path: Path) -> None:
    report = solve_selection(problem(), small_config())
    path = tmp_path / "receipt.json"
    save_receipt(report, path)
    restored = restore_receipt(path)
    assert restored.receipt_sha256 == report.receipt_sha256
    assert restored.deterministic_sha256 == report.deterministic_sha256
    assert restored.trials == report.trials
    assert restored.weights == report.weights
    with pytest.raises(FileExistsError):
        save_receipt(report, path)
    payload = json.loads(path.read_text())
    payload["body"]["trials"][0]["accepted"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        restore_receipt(path)


@pytest.mark.parametrize(
    "change",
    [
        "acceptance",
        "honesty",
        "future_clock",
        "derived_matrix",
        "schema",
        "extra_field",
        "missing_trial",
    ],
)
def test_rehashed_forgery_and_restore_schema_fail_closed(tmp_path: Path, change: str) -> None:
    report = solve_selection(problem(), small_config())
    path = tmp_path / "receipt.json"
    save_receipt(report, path)
    payload = json.loads(path.read_text())
    body = payload["body"]
    if change == "acceptance":
        trial = body["trials"][0]
        trial["accepted"] = max(trial["accepted_uphill"], trial["accepted"] - 1)
        if trial["accepted"] == report.trials[0].accepted:
            trial["accepted"] += 1
    elif change == "honesty":
        body["quantum_advantage"] = True
    elif change == "future_clock":
        body["problem"]["forecast"]["available_time"] = (NOW + timedelta(hours=1)).isoformat()
    elif change == "derived_matrix":
        body["qubo"]["matrix"][0][0] += 0.01
    elif change == "schema":
        body["schema"] = "claimed_device_v99"
    elif change == "extra_field":
        body["undeclared_source"] = "omitted from immutable source schema"
    else:
        body["trials"].pop()
    rehash(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        restore_receipt(path)


def test_malformed_and_resource_bound_restores_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    for content in ("{}", "[]", '{"body":null}', "not JSON"):
        path.write_text(content)
        with pytest.raises(ValueError):
            restore_receipt(path)
    path.write_text(" " * 8_000_001)
    with pytest.raises(ValueError, match="resource bound"):
        restore_receipt(path)
