"""Single-session convex allocation in NAV fractions; no silent risk relaxation.

See docs/COST_AWARE_CONSTRUCTION.md for units, assumptions and research basis.
This module accepts forecasts; it does not establish that they predict returns.

Solving is fail-closed across a fixed multi-solver chain (see ``SOLVER_CHAIN``)
with explicit status normalization (see ``normalize_status``): only a genuinely
optimal, constraint-satisfying solution may ever supply weights. Every attempt
is logged with solver name, normalized status, iteration count and wall-clock
(the hashed allocations ledger strips wall-clock so receipts stay
deterministic; pass ``attempt_log`` to capture full records).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationConfig:
    risk_window: int = 60
    risk_aversion: float = 20.0
    uncertainty_aversion: float = 1.0
    covariance_shrinkage: float = 0.25
    alpha_scale: float = 1.0
    turnover_limit: float = 0.25

    def validate(self) -> None:
        if type(self.risk_window) is not int or not 20 <= self.risk_window <= 252:
            raise ValueError("risk_window must be an integer in [20, 252]")
        for key in ("risk_aversion", "uncertainty_aversion", "alpha_scale", "turnover_limit"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.risk_aversion <= 0 or not 0 < self.turnover_limit <= 4:
            raise ValueError("risk_aversion > 0 and turnover_limit in (0, 4] required")
        if (
            isinstance(self.covariance_shrinkage, bool)
            or not np.isfinite(self.covariance_shrinkage)
            or not 0 <= self.covariance_shrinkage <= 1
        ):
            raise ValueError("covariance_shrinkage must be in [0, 1]")


class AllocationFailure(ValueError):
    """A rejected solve with diagnostics; it never provides tradable weights."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Status normalization: one dispatch table, fail-closed by default.
# ---------------------------------------------------------------------------

SELECTABLE_VERDICT = "selectable"
NOT_SELECTABLE_VERDICT = "not_selectable"

#: Canonical statuses that make a solve attempt permanently non-selectable.
STATUS_VERDICTS: dict[str, str] = {
    "optimal": SELECTABLE_VERDICT,
    "optimal_inaccurate": NOT_SELECTABLE_VERDICT,
    "suboptimal": NOT_SELECTABLE_VERDICT,
    "feasible": NOT_SELECTABLE_VERDICT,
    "infeasible": NOT_SELECTABLE_VERDICT,
    "infeasible_inaccurate": NOT_SELECTABLE_VERDICT,
    "unbounded": NOT_SELECTABLE_VERDICT,
    "unbounded_inaccurate": NOT_SELECTABLE_VERDICT,
    "numerical_error": NOT_SELECTABLE_VERDICT,
    "solver_error": NOT_SELECTABLE_VERDICT,
    "user_limit": NOT_SELECTABLE_VERDICT,
    "indeterminate": NOT_SELECTABLE_VERDICT,
    "not_attempted_unsupported_cone": NOT_SELECTABLE_VERDICT,
    "unknown": NOT_SELECTABLE_VERDICT,
}

#: Raw solver strings mapped to canonical statuses; anything else -> "unknown".
_RAW_STATUS_ALIASES: dict[str, str] = {
    "optimal": "optimal",
    "solved": "optimal",
    "optimal_inaccurate": "optimal_inaccurate",
    "solved_inaccurate": "optimal_inaccurate",
    "solved inaccurate": "optimal_inaccurate",
    "almost_solved": "optimal_inaccurate",
    "suboptimal": "suboptimal",
    "feasible": "feasible",
    "infeasible": "infeasible",
    "primal_infeasible": "infeasible",
    "primal infeasible": "infeasible",
    "infeasible_inaccurate": "infeasible_inaccurate",
    "primal_infeasible_inaccurate": "infeasible_inaccurate",
    "unbounded": "unbounded",
    "dual_infeasible": "unbounded",
    "dual infeasible": "unbounded",
    "unbounded_inaccurate": "unbounded_inaccurate",
    "dual_infeasible_inaccurate": "unbounded_inaccurate",
    "numerical_error": "numerical_error",
    "solver_error": "solver_error",
    "failed": "solver_error",
    "user_limit": "user_limit",
    "maximum_iterations_reached": "user_limit",
    "maximum iterations reached": "user_limit",
    "iteration_limit_reached": "user_limit",
    "time_limit_reached": "user_limit",
    "indeterminate": "indeterminate",
    "unsolved": "indeterminate",
    "not_attempted_unsupported_cone": "not_attempted_unsupported_cone",
}

#: Formulation-level statuses that terminate the whole allocation as failed.
TERMINAL_INFEASIBLE_STATUSES = frozenset({"infeasible", "unbounded"})


def normalize_status(raw: str | None) -> tuple[str, str]:
    """Map any solver-reported status to ``(canonical, verdict)``.

    Only ``optimal`` is selectable-eligible; every other or unknown status is
    a diagnostic and can never supply weights.
    """
    key = (raw or "").strip().lower()
    canonical = _RAW_STATUS_ALIASES.get(key, "unknown")
    verdict = STATUS_VERDICTS.get(canonical, NOT_SELECTABLE_VERDICT)
    return canonical, verdict


# ---------------------------------------------------------------------------
# Solver chain: fixed order, capability-gated, every attempt recorded.
# ---------------------------------------------------------------------------

SOLVER_CHAIN: tuple[str, ...] = ("CLARABEL", "OSQP", "SCS", "HIGHS")

#: Cones each solver can represent exactly. ECOS is intentionally absent: it
#: is not an installed dependency and must not be added silently.
SOLVER_CONE_SUPPORT: dict[str, frozenset[str]] = {
    "CLARABEL": frozenset({"linear", "quadratic", "power"}),
    "OSQP": frozenset({"linear", "quadratic"}),
    "SCS": frozenset({"linear", "quadratic", "power"}),
    "HIGHS": frozenset({"linear"}),
}

#: Target accuracy of the objective gap in ORIGINAL objective units. This is a
#: uniform numerical bar for every formulation, solver and input scaling: the
#: September protocol effectively demanded 1e-10 on scaled formulations and
#: 1e-8 on unscaled ones, which is what produced `optimal_inaccurate` outliers.
#: 1e-10 is the strictest historical demand, applied everywhere.
GAP_TOL_ORIGINAL = 1e-10
#: Feasibility accuracy in original NAV-fraction units.
FEAS_TOL_ORIGINAL = 1e-8


def problem_cones(has_impact: bool) -> frozenset[str]:
    """Cones the exact model needs: quadratic risk always, 3/2-power impact if used."""
    base = ("linear", "quadratic")
    return frozenset((*base, "power")) if has_impact else frozenset(base)


def solver_supports(solver: str, cones: frozenset[str]) -> bool:
    return cones <= SOLVER_CONE_SUPPORT.get(solver, frozenset())


def solver_tolerances(solver: str, *, gap_abs: float) -> dict[str, float]:
    """Solver accuracy knobs with the SAME meaning across solvers.

    ``gap_abs`` is the objective-gap tolerance in the units presented to the
    solver (see ``_objective_scale``); feasibility stays in NAV fractions.
    """
    dispatch: dict[str, dict[str, float]] = {
        "CLARABEL": {
            "tol_gap_abs": gap_abs,
            "tol_gap_rel": GAP_TOL_ORIGINAL,
            "tol_feas": FEAS_TOL_ORIGINAL,
        },
        "OSQP": {"eps_abs": gap_abs, "eps_rel": GAP_TOL_ORIGINAL},
        "SCS": {"eps_abs": gap_abs, "eps_rel": GAP_TOL_ORIGINAL},
        "HIGHS": {},
    }
    return dict(dispatch.get(solver, {}))


def _objective_scale(
    alpha: np.ndarray,
    uncertainty: np.ndarray,
    eigen_max: float,
    impact: np.ndarray,
    config: AllocationConfig,
    linear_cost: float,
    borrow_cost: float,
    funding_cost: float,
    cash_return: float,
) -> float:
    """Deterministic uniform objective scale: 1 / largest objective coefficient.

    The transform is exactly equivalent (uniform positive scaling of the
    objective leaves the argmax unchanged) and homogeneous of degree one in
    the economic inputs, so differently scaled copies of the same problem are
    presented to the solver identically.
    """
    impact_max = float(np.max(impact)) if impact.size else 0.0
    coefficients = [
        float(np.max(np.abs(alpha))) if alpha.size else 0.0,
        config.uncertainty_aversion * float(np.max(uncertainty)) if uncertainty.size else 0.0,
        config.risk_aversion * float(max(eigen_max, 0.0)),
        config.uncertainty_aversion * 0.0,
        float(linear_cost),
        impact_max,
        float(borrow_cost),
        float(funding_cost - cash_return),
        float(cash_return),
    ]
    magnitude = max(coefficients)
    return 1.0 / magnitude if magnitude > 0 else 1.0


def _cost_unit(
    linear_cost: float, impact: np.ndarray, turnover_limit: float, epigraph_boost: float
) -> float:
    """Deterministic cost-epigraph unit: the worst one-session traded cost.

    The epigraph variable is expressed in these units so it is O(1) instead of
    an O(1e-4) dust variable; the substitution is exactly equivalent and
    homogeneous of degree one (the unit is, the scale is degree zero).
    """
    impact_max = float(np.max(impact)) if impact.size else 0.0
    worst = linear_cost * turnover_limit + impact_max * turnover_limit**1.5
    return worst / epigraph_boost if worst > 0 else 1.0 / epigraph_boost


@dataclass(frozen=True)
class _Formulation:
    """Exactly equivalent conic representation; only conditioning differs."""

    name: str
    bound_capacity: bool
    factor_risk: bool
    epigraph_boost: float


FORMULATIONS: tuple[_Formulation, ...] = (
    _Formulation("original", False, False, 1.0),
    _Formulation("bounded_capacity", True, False, 1.0),
    _Formulation("factored_risk", True, True, 1.0),
    _Formulation("scaled_cost", True, True, 10_000.0),
)


def _strip_timing(record: dict[str, Any]) -> dict[str, Any]:
    """Wall-clock stays out of hashed ledgers so receipts remain deterministic."""
    cleaned = {k: v for k, v in record.items() if k != "solve_time_seconds"}
    nested = cleaned.get("solver_attempts")
    if isinstance(nested, list):
        cleaned["solver_attempts"] = [
            {k: v for k, v in item.items() if k != "solve_time_seconds"} for item in nested
        ]
    return cleaned


def _stats(problem: cp.Problem) -> tuple[int | None, float | None]:
    stats = problem.solver_stats
    iterations = getattr(stats, "num_iters", None) if stats is not None else None
    elapsed = getattr(stats, "solve_time", None) if stats is not None else None
    return (
        iterations if type(iterations) is int and iterations >= 0 else None,
        float(elapsed)
        if isinstance(elapsed, (int, float)) and np.isfinite(elapsed) and elapsed >= 0
        else None,
    )


def _attempt_record(
    problem: cp.Problem, *, formulation: str, solver: str, raw_status: str | None
) -> dict[str, Any]:
    canonical, verdict = normalize_status(raw_status)
    iterations, elapsed = _stats(problem)
    return {
        "solver": solver,
        "formulation": formulation,
        "raw_status": raw_status,
        "status": canonical,
        "selection_verdict": verdict,
        "iterations": iterations,
        "solve_time_seconds": elapsed,
        "weights_accepted": False,
    }


def _candidate_checks(
    weights: np.ndarray,
    *,
    alpha: np.ndarray,
    covariance: np.ndarray,
    uncertainty: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: np.ndarray,
    impact: np.ndarray,
    config: AllocationConfig,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float,
    funding_cost: float,
    cash_return: float,
) -> tuple[float, float, float, float]:
    """Check the original unscaled economics and limits, independent of CVXPY."""
    change = np.abs(weights - previous)
    cost = float(linear_cost * change.sum() + impact @ change**1.5)
    holding = float(
        borrow_cost * np.maximum(-weights, 0).sum()
        + cash_return * weights.sum()
        + (funding_cost - cash_return) * max(float(weights.sum()) - 1, 0)
    )
    objective = float(
        alpha @ weights
        - config.risk_aversion * (weights @ covariance @ weights)
        - config.uncertainty_aversion * (uncertainty @ np.abs(weights))
        - cost
        - holding
    )
    violations = [
        float(np.max(lower - weights)),
        float(np.max(weights - upper)),
        float(np.max(change - capacity)),
        float(change.sum() - config.turnover_limit),
        float(np.abs(weights).sum() + gross_limit * cost - gross_limit),
        float(np.max(np.abs(weights) + name_limit * cost - name_limit)),
    ]
    if gross_limit <= 1:
        violations.append(float(weights.sum() + cost - (1 - cash_buffer)))
    return max(0.0, *violations), objective, cost, holding


def _build_conic_problem(
    formulation: _Formulation,
    a: np.ndarray,
    cov: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    u: np.ndarray,
    prev: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cap: np.ndarray,
    imp: np.ndarray,
    config: AllocationConfig,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float,
    funding_cost: float,
    cash_return: float,
    objective_scale: float,
    cost_unit: float,
) -> tuple[cp.Problem, cp.Variable, list[Constraint]]:
    """One exactly equivalent conic form of the allocation problem."""
    w = cp.Variable(a.size)
    delta = cp.Variable(a.size, nonneg=True)
    trade_cost_variable = cp.Variable(nonneg=True)
    estimated_cost = linear_cost * cp.sum(delta) + (
        imp[imp > 0] @ cp.power(delta[imp > 0], 1.5) if np.any(imp > 0) else cp.Constant(0)
    )
    if formulation.factor_risk:
        risk_factor = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
        risk = cp.sum_squares(risk_factor @ w)
    else:
        risk = cp.quad_form(w, cp.psd_wrap(cov))
    holding_cost = (
        borrow_cost * cp.sum(cp.pos(-w))
        + cash_return * cp.sum(w)
        + (funding_cost - cash_return) * cp.pos(cp.sum(w) - 1)
    )
    objective = (
        a @ w
        - config.risk_aversion * risk
        - config.uncertainty_aversion * (u @ cp.abs(w))
        - trade_cost_variable * cost_unit
        - holding_cost
    )
    # Since delta >= 0 and sum(delta) <= turnover_limit, min(cap, turnover)
    # removes redundant huge NAV-fraction bounds exactly.
    capacity_bound = np.minimum(cap, config.turnover_limit)
    if not formulation.bound_capacity:
        capacity_bound = cap
    constraints = [
        delta >= w - prev,
        delta >= prev - w,
        trade_cost_variable >= estimated_cost / cost_unit,
        w >= lo,
        w <= hi,
        delta <= capacity_bound,
        cp.sum(delta) <= config.turnover_limit,
        cp.norm1(w) + gross_limit * trade_cost_variable * cost_unit <= gross_limit,
        cp.abs(w) + name_limit * trade_cost_variable * cost_unit <= name_limit,
    ]
    if gross_limit <= 1:
        constraints.append(cp.sum(w) + trade_cost_variable * cost_unit <= 1 - cash_buffer)
    problem = cp.Problem(cp.Maximize(objective_scale * objective), constraints)
    return problem, w, constraints


def _check_solution(
    problem: cp.Problem,
    w: cp.Variable,
    constraints: list[Any],
    *,
    formulation: _Formulation,
    solver: str,
    objective_scale: float,
    cost_unit: float,
    previous: np.ndarray,
    alpha: np.ndarray,
    covariance: np.ndarray,
    uncertainty: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: np.ndarray,
    impact: np.ndarray,
    config: AllocationConfig,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float,
    funding_cost: float,
    cash_return: float,
) -> tuple[dict[str, Any], tuple[np.ndarray, float, float, float, float] | None]:
    """Independent original-unit recheck before any weights may be accepted."""
    record = _attempt_record(
        problem, formulation=formulation.name, solver=solver, raw_status=str(problem.status)
    )
    if record["selection_verdict"] != SELECTABLE_VERDICT:
        # Fail-closed status gate: only a genuinely optimal solve may even be
        # checked for selection; everything else stays a diagnostic.
        return record, None
    if w.value is None or problem.value is None:
        record["status"] = "nonfinite_solution" if w.value is None else "solver_error"
        record["selection_verdict"] = NOT_SELECTABLE_VERDICT
        return record, None
    result = np.asarray(w.value, dtype=float).reshape(-1).copy()
    if not np.isfinite(result).all():
        record["status"] = "nonfinite_solution"
        record["selection_verdict"] = NOT_SELECTABLE_VERDICT
        return record, None
    solver_objective = float(problem.value) / objective_scale
    # Keep the original no-trade dust policy, then verify both the model
    # constraints and the original unscaled formulas before acceptance.
    result[np.abs(result - previous) < 1e-8] = previous[np.abs(result - previous) < 1e-8]
    result[np.abs(result) < 1e-10] = 0.0
    primal, exact_objective, actual_cost, actual_holding = _candidate_checks(
        result,
        alpha=alpha,
        covariance=covariance,
        uncertainty=uncertainty,
        previous=previous,
        lower=lower,
        upper=upper,
        capacity=capacity,
        impact=impact,
        config=config,
        linear_cost=linear_cost,
        gross_limit=gross_limit,
        name_limit=name_limit,
        cash_buffer=cash_buffer,
        borrow_cost=borrow_cost,
        funding_cost=funding_cost,
        cash_return=cash_return,
    )
    w.value = result
    conic = max(float(np.max(c.violation())) for c in constraints)
    violation = max(primal, conic)
    objective_gap = abs(solver_objective - exact_objective)
    record.update({"max_constraint_violation": violation, "solver_objective_gap": objective_gap})
    certifiable = (
        np.isfinite([violation, exact_objective, objective_gap]).all()
        and violation <= 1e-7
        and objective_gap <= 1e-7
    )
    if certifiable:
        record["weights_accepted"] = True
        payload = (result, violation, exact_objective, actual_cost, actual_holding)
        return record, payload
    record["status"] = "independent_check_failed"
    record["selection_verdict"] = NOT_SELECTABLE_VERDICT
    return record, None


_INFEASIBILITY_FAMILY = frozenset(
    {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}
)


def _aggregate_status(records: list[dict[str, Any]]) -> str:
    """One representative formulation status from its solver-attempt records."""
    statuses = [r["status"] for r in records if r["status"] != "not_attempted_unsupported_cone"]
    if not statuses:
        return "not_attempted_unsupported_cone"
    if any(record["weights_accepted"] for record in records):
        return "optimal"
    if "independent_check_failed" in statuses:
        return "independent_check_failed"
    if all(status in _INFEASIBILITY_FAMILY for status in statuses):
        certified = [s for s in ("infeasible", "unbounded") if s in statuses]
        return str(certified[0] if certified else statuses[0])
    if all(status == "solver_error" for status in statuses):
        return "solver_error"
    return str(statuses[-1])


def _log_attempt(record: dict[str, Any]) -> None:
    logger.info(
        "allocation attempt solver=%s formulation=%s status=%s iterations=%s wall_clock_seconds=%s",
        record["solver"],
        record["formulation"],
        record["status"],
        record["iterations"],
        record["solve_time_seconds"],
    )


def _run_solver_chain(
    formulation: _Formulation,
    cones: frozenset[str],
    *,
    gap_abs: float,
    attempt_log: list[dict[str, Any]] | None,
    build: Any,
    verify: Any,
) -> tuple[dict[str, Any], tuple[np.ndarray, float, float, float, float] | None]:
    """Try each chain solver on a fresh problem; only certified optimal passes."""
    records: list[dict[str, Any]] = []
    payload: tuple[np.ndarray, float, float, float, float] | None = None
    for solver in SOLVER_CHAIN:
        if not solver_supports(solver, cones):
            records.append(
                {
                    "solver": solver,
                    "formulation": formulation.name,
                    "raw_status": "not_attempted_unsupported_cone",
                    "status": "not_attempted_unsupported_cone",
                    "selection_verdict": NOT_SELECTABLE_VERDICT,
                    "iterations": None,
                    "solve_time_seconds": None,
                    "weights_accepted": False,
                }
            )
            continue
        problem, w, constraints = build()
        tolerances = solver_tolerances(solver, gap_abs=gap_abs)
        try:
            problem.solve(solver=solver, **tolerances)
        except cp.error.SolverError as exc:
            record = _attempt_record(
                problem, formulation=formulation.name, solver=solver, raw_status="solver_error"
            )
            record["error"] = str(exc)
            records.append(record)
            _log_attempt(record)
            if attempt_log is not None:
                attempt_log.append(dict(record))
            continue
        record, solved = verify(problem, w, constraints)
        record["solver"] = solver
        record["formulation"] = formulation.name
        records.append(record)
        _log_attempt(record)
        if attempt_log is not None:
            attempt_log.append(dict(record))
        if solved is not None:
            payload = solved
            break
    summary = {
        "formulation": formulation.name,
        "status": _aggregate_status(records),
        "iterations": records[-1]["iterations"] if records else None,
        "solve_time_seconds": None,
        "weights_accepted": payload is not None,
        "solver_attempts": records,
    }
    return summary, payload


def allocate(
    alpha: np.ndarray,
    covariance: np.ndarray,
    uncertainty: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: np.ndarray,
    impact: np.ndarray,
    *,
    config: AllocationConfig,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float = 0.0,
    funding_cost: float = 0.0,
    cash_return: float = 0.0,
    attempt_log: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize from actual holdings, with costs and capacity normalized by NAV.

    impact_i * |delta_weight_i|**1.5 is the estimated impact/NAV. Holding
    rates and forecasts are for one planning session. All inputs are as-of.
    """
    config.validate()
    inputs = _validated_inputs(
        alpha,
        covariance,
        uncertainty,
        previous,
        lower,
        upper,
        capacity,
        impact,
        linear_cost=linear_cost,
        gross_limit=gross_limit,
        name_limit=name_limit,
        cash_buffer=cash_buffer,
        borrow_cost=borrow_cost,
        funding_cost=funding_cost,
        cash_return=cash_return,
    )
    solved = _solve_ladder(inputs, config, attempt_log)
    return _finalize(solved, inputs, config)


def _validated_inputs(
    alpha: np.ndarray,
    covariance: np.ndarray,
    uncertainty: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: np.ndarray,
    impact: np.ndarray,
    *,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float,
    funding_cost: float,
    cash_return: float,
) -> dict[str, Any]:
    """Fail-closed input validation and deterministic PSD repair."""
    a, cov = np.asarray(alpha, dtype=float), np.asarray(covariance, dtype=float)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("alpha must be a nonempty finite vector")
    vectors = [
        np.asarray(v, dtype=float) for v in (uncertainty, previous, lower, upper, capacity, impact)
    ]
    if any(v.shape != a.shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError("allocation vectors must be finite and match alpha")
    u, prev, lo, hi, cap, imp = vectors
    if np.any(u < 0) or np.any(cap < 0) or np.any(imp < 0) or np.any(lo > hi):
        raise ValueError("invalid uncertainty, capacity, impact or bounds")
    if (
        cov.shape != (a.size, a.size)
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T, atol=1e-12, rtol=0)
    ):
        raise ValueError("covariance must be finite, square and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    cov = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    rates = [linear_cost, borrow_cost, funding_cost, cash_return]
    if not np.isfinite(rates).all() or min(rates) < 0 or funding_cost < cash_return:
        raise ValueError("nonnegative costs and funding_cost >= cash_return required")
    if (
        not np.isfinite([gross_limit, name_limit, cash_buffer]).all()
        or not 0 < gross_limit <= 2
        or not 0 < name_limit <= 1
        or not 0 <= cash_buffer < 1
    ):
        raise ValueError("invalid exposure or cash limits")
    return {
        "a": a,
        "cov": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "u": u,
        "prev": prev,
        "lo": lo,
        "hi": hi,
        "cap": cap,
        "imp": imp,
        "linear_cost": linear_cost,
        "gross_limit": gross_limit,
        "name_limit": name_limit,
        "cash_buffer": cash_buffer,
        "borrow_cost": borrow_cost,
        "funding_cost": funding_cost,
        "cash_return": cash_return,
    }


def _solve_ladder(
    inputs: dict[str, Any], config: AllocationConfig, attempt_log: list[dict[str, Any]] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every formulation through the solver chain until one is certified."""
    a, u, imp = inputs["a"], inputs["u"], inputs["imp"]
    scale = _objective_scale(
        a,
        u,
        float(np.max(inputs["eigenvalues"])),
        imp,
        config,
        inputs["linear_cost"],
        inputs["borrow_cost"],
        inputs["funding_cost"],
        inputs["cash_return"],
    )
    gap_abs = GAP_TOL_ORIGINAL * scale
    cones = problem_cones(bool(np.any(imp > 0)))
    attempts: list[dict[str, Any]] = []
    for formulation in FORMULATIONS:
        cost_unit = _cost_unit(
            inputs["linear_cost"], imp, config.turnover_limit, formulation.epigraph_boost
        )
        summary, payload = _run_solver_chain(
            formulation,
            cones,
            gap_abs=gap_abs,
            attempt_log=attempt_log,
            build=lambda f=formulation, c=cost_unit, s=scale: _build_conic_problem(
                f,
                a,
                inputs["cov"],
                inputs["eigenvalues"],
                inputs["eigenvectors"],
                u,
                inputs["prev"],
                inputs["lo"],
                inputs["hi"],
                inputs["cap"],
                imp,
                config,
                inputs["linear_cost"],
                inputs["gross_limit"],
                inputs["name_limit"],
                inputs["cash_buffer"],
                inputs["borrow_cost"],
                inputs["funding_cost"],
                inputs["cash_return"],
                s,
                c,
            ),
            verify=lambda p, w, cs, f=formulation, c=cost_unit, s=scale: _check_solution(
                p,
                w,
                cs,
                formulation=f,
                solver="",
                objective_scale=s,
                cost_unit=c,
                previous=inputs["prev"],
                alpha=a,
                covariance=inputs["cov"],
                uncertainty=u,
                lower=inputs["lo"],
                upper=inputs["hi"],
                capacity=inputs["cap"],
                impact=imp,
                config=config,
                linear_cost=inputs["linear_cost"],
                gross_limit=inputs["gross_limit"],
                name_limit=inputs["name_limit"],
                cash_buffer=inputs["cash_buffer"],
                borrow_cost=inputs["borrow_cost"],
                funding_cost=inputs["funding_cost"],
                cash_return=inputs["cash_return"],
            ),
        )
        attempts.append(summary)
        if payload is not None:
            return {"attempts": attempts, "formulation": formulation.name}, {
                "payload": payload,
                "objective_scale": scale,
                "cost_unit": cost_unit,
            }
        if summary["status"] in TERMINAL_INFEASIBLE_STATUSES:
            raise AllocationFailure(f"allocation failed: {summary['status']}", summary)
    failure = {**attempts[-1], "attempts": attempts, "weights_accepted": False}
    if all(attempt["status"] == "solver_error" for attempt in attempts):
        raise AllocationFailure("allocation solver failed: no accepted solution", failure)
    raise AllocationFailure("allocation failed: no accepted solution", failure)


def _finalize(
    solved: tuple[dict[str, Any], dict[str, Any]], inputs: dict[str, Any], config: AllocationConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic accepted-weights ledger in original units."""
    ladder, extra = solved
    result, violation, exact_objective, actual_cost, actual_holding = extra["payload"]
    prev, a, cov, u = inputs["prev"], inputs["a"], inputs["cov"], inputs["u"]
    imp, linear_cost = inputs["imp"], inputs["linear_cost"]
    d = np.abs(result - prev)
    attempts = [_strip_timing(attempt) for attempt in ladder["attempts"]]
    return result, {
        "status": cp.OPTIMAL,
        "solver": attempts[-1]["solver_attempts"][-1]["solver"],
        "formulation": ladder["formulation"],
        "solve_attempts": attempts,
        "objective": exact_objective,
        "expected_return_proxy": float(a @ result),
        "predicted_variance": float(result @ cov @ result),
        "uncertainty_penalty": float(config.uncertainty_aversion * (u @ np.abs(result))),
        "predicted_linear_cost": float(linear_cost * d.sum()),
        "predicted_impact_cost": float(imp @ d**1.5),
        "predicted_holding_cost": actual_holding,
        "turnover": float(d.sum()),
        "max_constraint_violation": violation,
        "previous_weights": prev.tolist(),
        "target_weights": result.tolist(),
        "alpha": a.tolist(),
        "uncertainty": u.tolist(),
        "capacity": inputs["cap"].tolist(),
    }
