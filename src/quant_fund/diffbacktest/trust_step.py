"""Trust-region stepping on the smooth surrogate, accepted by the hard book.

``grad_fidelity`` measured that the smooth surrogate's gradient is
directionally right in aggregate (high cosine) but invents per-knob slope
inside flat discrete regions (~44% phantom share). The usable consequence:
step along the joint surrogate direction, but *accept* only when the
hardened book's objective actually improves — classic trust-region:

- step ``θ⁺ = project(θ + r·ĝ)`` on the box
- ``hard(θ⁺) > hard(θ)`` → accept, radius × grow
- else → reject, radius × shrink

``trust_optimize`` runs that loop and reports the achieved hard objective
against a naive surrogate-ascent baseline (same budget, no acceptance
check) — the direct test of whether the acceptance gate buys anything on
a piecewise-flat objective.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.diffbacktest.numpy_core import (
    objective_value,
    simulate,
    synthetic_prices,
    warmup_start,
)
from quant_fund.diffbacktest.spec import (
    BOXES,
    StrategyParams,
    active_parameters,
    harden,
    pack,
    project_box,
    unpack,
    validate_params,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def _feasible_sample(
    rng: np.random.Generator, names: tuple[str, ...], base: StrategyParams, strategy: str
) -> StrategyParams:
    lo = np.array([BOXES[n][0] for n in names])
    hi = np.array([BOXES[n][1] for n in names])
    for _ in range(64):
        cand = project_box(unpack(rng.uniform(lo, hi), names, base), names)
        try:
            validate_params(strategy, cand)
        except ValueError:
            continue
        return cand
    return base  # fall back to defaults — validate_params passes for defaults


def _hard_objective(
    prices: np.ndarray, strategy: str, params: StrategyParams, objective: str
) -> float:
    try:
        sim = simulate(prices, strategy, harden(params))
    except ValueError:
        return float("nan")
    return objective_value(
        sim.net,
        objective,
        periods_per_year=params.periods_per_year,
        score_start=warmup_start(strategy, params),
    )


def _smooth_gradient(
    prices: np.ndarray, strategy: str, params: StrategyParams, objective: str
) -> tuple[Array, tuple[str, ...]]:
    from quant_fund.diffbacktest.jax_core import available, objective_gradients

    names = active_parameters(strategy)
    if not available() or not names:
        return np.zeros(len(names)), names
    res = objective_gradients(prices, strategy, params, objective, mode="smooth")
    g = np.asarray(res.parameter_gradient, dtype=float)
    return np.where(np.isfinite(g), g, 0.0), tuple(res.parameter_names)


def trust_optimize(
    prices: np.ndarray,
    strategy: str,
    *,
    objective: str = "sharpe",
    init_params: StrategyParams | None = None,
    radius0: float = 0.15,
    grow: float = 1.5,
    shrink: float = 0.5,
    n_iters: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Trust-region ascent on the hard objective via surrogate directions."""
    base = init_params or StrategyParams()
    names = active_parameters(strategy)
    if not names:
        raise ValueError(f"{strategy}: no active parameters")
    lo = np.array([BOXES[n][0] for n in names])
    hi = np.array([BOXES[n][1] for n in names])
    rng = np.random.default_rng(seed)
    start = _feasible_sample(rng, names, base, strategy)

    def run(accept_gate: bool) -> dict[str, Any]:
        params = start
        theta = pack(params, names)
        cur = _hard_objective(prices, strategy, params, objective)
        radius = radius0
        n_accept = 0
        n_reject = 0
        n_infeasible = 0
        history = [cur]
        for _ in range(n_iters):
            g, grad_names = _smooth_gradient(prices, strategy, params, objective)
            if list(grad_names) != list(names) or not np.any(g):
                break
            norm = np.linalg.norm(g)
            step = g * (radius / norm)
            theta_prop = np.clip(theta + step, lo, hi)
            cand = project_box(unpack(theta_prop, names, params), names)
            try:
                validate_params(strategy, cand)
            except ValueError:
                n_infeasible += 1
                radius *= shrink
                history.append(cur)
                continue
            val = _hard_objective(prices, strategy, cand, objective)
            if np.isnan(val):
                n_infeasible += 1
                radius *= shrink
                history.append(cur)
                continue
            if val > cur or not accept_gate:
                params, theta, cur = cand, theta_prop, val
                n_accept += 1
                radius = min(radius * grow, 2.0) if accept_gate else radius
            else:
                n_reject += 1
                radius *= shrink
            history.append(cur)
        return {
            "final_objective": cur,
            "history": history,
            "n_accept": n_accept,
            "n_reject": n_reject,
            "n_infeasible": n_infeasible,
        }

    initial = _hard_objective(prices, strategy, start, objective)
    gated = run(accept_gate=True)
    naive = run(accept_gate=False)
    return {
        "initial_objective": initial,
        "gated": gated,
        "naive": naive,
        "gate_helps": gated["final_objective"] >= naive["final_objective"],
        "improvement_gated": (gated["final_objective"] - initial if np.isfinite(initial) else None),
        "improvement_naive": (naive["final_objective"] - initial if np.isfinite(initial) else None),
    }


def trust_bench(
    seed: int = 0, n_steps: int = 400, n_names: int = 6, n_seeds: int = 4
) -> dict[str, Any]:
    """Gated vs naive ascent over seeded panels; sealed receipt.

    One seed is a coin flip — report the share where the acceptance gate
    helps, plus mean deltas.
    """
    runs = []
    for s in range(n_seeds):
        prices = synthetic_prices(n_steps=n_steps, n_names=n_names, seed=seed + s)
        runs.append(trust_optimize(prices, "tsmom", objective="sharpe", seed=seed + s))
    helps = sum(1 for r in runs if r["gate_helps"])
    deltas = [
        r["gated"]["final_objective"] - r["naive"]["final_objective"]
        for r in runs
        if np.isfinite(r["gated"]["final_objective"]) and np.isfinite(r["naive"]["final_objective"])
    ]

    def _finite(x: Any) -> Any:
        return None if isinstance(x, float) and not np.isfinite(x) else x

    def _scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_scrub(v) for v in obj]
        return _finite(obj)

    payload: dict[str, Any] = {
        "kind": "trust_step",
        "schema": "trust_step.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "hard-book acceptance gate filters phantom surrogate steps",
            "gate_helps_share": helps / n_seeds,
            "mean_gated_minus_naive": float(np.mean(deltas)) if deltas else None,
            "verdict": "ok" if helps > n_seeds // 2 else "weak",
        },
        "interpretation": {"runs": [_scrub(r) for r in runs]},
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
