"""Fail-closed solver-chain and conditioning checks on SYNTHETIC inputs.

These tests verify numerical machinery only. They are correctness tests, never
market evidence, and they carry no live-trading or profitability claim.
"""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from quant_fund.research import cost_allocation as ca
from quant_fund.research.cost_allocation import (
    SELECTABLE_VERDICT,
    STATUS_VERDICTS,
    AllocationConfig,
    allocate,
    normalize_status,
    solver_supports,
)


def _problem(**overrides):
    fields = dict(
        alpha=np.array([0.01, 0.004, -0.002]),
        covariance=np.array(
            [[0.001, 0.0002, 0.0], [0.0002, 0.0012, 0.0001], [0.0, 0.0001, 0.0009]]
        ),
        uncertainty=np.array([0.001, 0.001, 0.001]),
        previous=np.zeros(3),
        lower=np.full(3, -0.3),
        upper=np.full(3, 0.3),
        capacity=np.ones(3),
        impact=np.zeros(3),
        config=AllocationConfig(uncertainty_aversion=0.5, turnover_limit=1.0),
        linear_cost=0.0006,
        gross_limit=0.9,
        name_limit=0.3,
        cash_buffer=0.05,
        borrow_cost=0.03 / 252,
        funding_cost=0.06 / 252,
        cash_return=0.0,
    )
    fields.update(overrides)
    return fields


def _scaled(fields, k):
    keys = (
        "alpha",
        "covariance",
        "uncertainty",
        "impact",
        "linear_cost",
        "borrow_cost",
        "funding_cost",
        "cash_return",
    )
    return {**fields, **{key: fields[key] * k for key in keys}}


def test_every_status_maps_to_exactly_one_verdict():
    for raw in STATUS_VERDICTS:
        canonical, verdict = normalize_status(raw)
        assert canonical == raw
        assert verdict in {SELECTABLE_VERDICT, ca.NOT_SELECTABLE_VERDICT}
    for raw in ca._RAW_STATUS_ALIASES:
        canonical, verdict = normalize_status(raw)
        assert canonical in STATUS_VERDICTS
        assert verdict == STATUS_VERDICTS[canonical]
    assert len(STATUS_VERDICTS) >= 14


def test_only_genuinely_optimal_status_is_selectable():
    for raw in STATUS_VERDICTS:
        _, verdict = normalize_status(raw)
        if raw == "optimal":
            assert verdict == SELECTABLE_VERDICT
        else:
            assert verdict == ca.NOT_SELECTABLE_VERDICT
    for raw in (
        "optimal_inaccurate",
        "suboptimal",
        "infeasible",
        "unbounded",
        "numerical_error",
        "solver_error",
    ):
        assert normalize_status(raw)[1] == ca.NOT_SELECTABLE_VERDICT


@pytest.mark.parametrize("raw", [None, "", "  ", "banana", "OPTIMALISH", "solvedish"])
def test_unknown_status_fails_closed(raw):
    assert normalize_status(raw) == ("unknown", ca.NOT_SELECTABLE_VERDICT)


@pytest.mark.parametrize(
    "forced",
    ["optimal_inaccurate", "suboptimal", "feasible", "user_limit", "indeterminate"],
)
def test_nonselectable_status_never_provides_weights(monkeypatch, forced):
    original = cp.Problem.solve

    def inaccurate(problem, *args, **kwargs):
        value = original(problem, *args, **kwargs)
        problem._status = forced
        return value

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    with pytest.raises(ca.AllocationFailure, match="no accepted solution") as failed:
        allocate(**_problem())
    attempts = failed.value.diagnostic["attempts"]
    assert len(attempts) == len(ca.FORMULATIONS)
    assert all(attempt["weights_accepted"] is False for attempt in attempts)
    assert all(
        attempt["selection_verdict"] == ca.NOT_SELECTABLE_VERDICT
        for attempt in attempts[0]["solver_attempts"]
    )


def test_constraint_violation_blocks_selection(monkeypatch):
    original = ca._candidate_checks

    def violating(*args, **kwargs):
        primal, objective, cost, holding = original(*args, **kwargs)
        return max(primal, 2e-7), objective, cost, holding

    monkeypatch.setattr(ca, "_candidate_checks", violating)
    with pytest.raises(ca.AllocationFailure, match="no accepted solution") as failed:
        allocate(**_problem())
    assert all(attempt["weights_accepted"] is False for attempt in failed.value.diagnostic["attempts"])
    assert all(
        attempt["status"] == "independent_check_failed"
        for attempt in failed.value.diagnostic["attempts"]
    )


def test_every_attempt_logs_solver_status_iterations_and_wall_clock():
    log = []
    _, diagnostic = allocate(**_problem(), attempt_log=log)
    assert log and diagnostic["status"] == cp.OPTIMAL
    for record in log:
        assert {"solver", "status", "iterations", "solve_time_seconds"} <= set(record)
        assert record["solver"] in ca.SOLVER_CHAIN
    assert log[0]["solver"] == "CLARABEL"
    assert log[0]["status"] == "optimal"
    # The hashed ledger stays deterministic: wall-clock is recorded but stripped.
    ledger = diagnostic["solve_attempts"]
    assert all("solve_time_seconds" not in attempt for attempt in ledger)
    assert all(
        "solve_time_seconds" not in solver_attempt
        for attempt in ledger
        for solver_attempt in attempt["solver_attempts"]
    )


def test_capability_gating_skips_ineligible_solvers(monkeypatch):
    assert solver_supports("CLARABEL", ca.problem_cones(True))
    assert solver_supports("SCS", ca.problem_cones(True))
    assert not solver_supports("OSQP", ca.problem_cones(True))  # 3/2-power impact cone
    assert not solver_supports("HIGHS", ca.problem_cones(True))
    assert not solver_supports("HIGHS", ca.problem_cones(False))  # quadratic risk

    def broken(*args, **kwargs):
        raise cp.error.SolverError("injected")

    monkeypatch.setattr(cp.Problem, "solve", broken)
    with pytest.raises(ca.AllocationFailure) as failed_power:
        allocate(**_problem(impact=np.full(3, 0.01)))
    with pytest.raises(ca.AllocationFailure) as failed_qp:
        allocate(**_problem())

    def chain_of(diagnostic):
        return {
            attempt["solver"]: attempt["status"]
            for attempt in diagnostic["attempts"][0]["solver_attempts"]
        }

    power_chain = chain_of(failed_power.value.diagnostic)
    assert set(power_chain) == set(ca.SOLVER_CHAIN)
    assert power_chain["CLARABEL"] == "solver_error"
    assert power_chain["OSQP"] == "not_attempted_unsupported_cone"  # power cone
    assert power_chain["SCS"] == "solver_error"
    assert power_chain["HIGHS"] == "not_attempted_unsupported_cone"  # quadratic risk

    qp_chain = chain_of(failed_qp.value.diagnostic)
    assert qp_chain["OSQP"] == "solver_error"  # impact=0: QP-capable, attempted
    assert qp_chain["HIGHS"] == "not_attempted_unsupported_cone"


def test_solver_fallback_selects_next_solver_in_chain(monkeypatch):
    original = cp.Problem.solve

    def broken(problem, *args, **kwargs):
        if kwargs.get("solver") == "CLARABEL":
            raise cp.error.SolverError("injected")
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", broken)
    _, diagnostic = allocate(**_problem())
    assert diagnostic["solver"] == "OSQP"
    chain = diagnostic["solve_attempts"][0]["solver_attempts"]
    assert [attempt["status"] for attempt in chain if attempt["solver"] == "CLARABEL"] == [
        "solver_error"
    ]


def test_all_solver_error_keeps_failure_label(monkeypatch):
    def broken(*args, **kwargs):
        raise cp.error.SolverError("injected")

    monkeypatch.setattr(cp.Problem, "solve", broken)
    with pytest.raises(ca.AllocationFailure, match="solver failed") as failed:
        allocate(**_problem())
    assert failed.value.diagnostic["status"] == "solver_error"
    assert failed.value.diagnostic["weights_accepted"] is False


@pytest.mark.parametrize("k", [1e-2, 1e-1, 1e1, 1e2])
def test_scaled_and_unscaled_problems_produce_the_same_solution(k):
    """Uniformly scaled copies of one problem are the same problem."""
    base = _problem(impact=np.full(3, 0.004))
    reference, _ = allocate(**base)
    scaled, _ = allocate(**_scaled(base, k))
    assert np.allclose(scaled, reference, atol=1e-6, rtol=0)


def test_every_exactly_equivalent_formulation_agrees(monkeypatch):
    """Different conic representations must not change the economic answer.

    The formulations are mathematically identical; measured agreement is
    ~3e-6 NAV fraction (solver stopping noise), asserted at 1e-5 (0.001%).
    """
    base = _problem(impact=np.full(3, 0.004))
    reference, _ = allocate(**base)
    ladder = ca.FORMULATIONS
    try:
        for formulation in ladder:
            monkeypatch.setattr(ca, "FORMULATIONS", (formulation,))
            weights, diagnostic = allocate(**base)
            assert diagnostic["formulation"] == formulation.name
            assert np.allclose(weights, reference, atol=1e-5, rtol=0)
    finally:
        monkeypatch.setattr(ca, "FORMULATIONS", ladder)


def test_market_derived_failure_fixture_is_certified_with_recorded_fallback():
    """A reconstructed September failure decision, as a numerical regression."""
    from pathlib import Path

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "cost_allocation"
        / "momentum_factor359.npz"
    )
    with np.load(fixture_path, allow_pickle=False) as fixture:
        fields = {key: fixture[key] for key in fixture.files}
    log = []
    weights, diagnostic = allocate(
        **fields,
        config=AllocationConfig(),
        linear_cost=0.0006,
        gross_limit=0.9405,
        name_limit=0.198,
        cash_buffer=0.01,
        borrow_cost=0.03 / 252,
        funding_cost=0.06 / 252,
        attempt_log=log,
    )
    assert diagnostic["status"] == cp.OPTIMAL
    assert diagnostic["max_constraint_violation"] <= 1e-7
    assert np.isfinite(weights).all()
    statuses = {record["status"] for record in log}
    assert "optimal_inaccurate" in statuses  # recorded as a diagnostic, never selected
    assert log[-1]["weights_accepted"] is True
    assert all(record["weights_accepted"] is False for record in log[:-1])


def test_config_validate_is_unchanged_fail_closed():
    with pytest.raises(ValueError):
        replace(AllocationConfig(), risk_aversion=0).validate()
    with pytest.raises(ValueError):
        replace(AllocationConfig(), turnover_limit=0).validate()
