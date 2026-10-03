"""Gradient-fidelity audit: does the smooth surrogate point uphill on the hard book?

The differentiable research book relaxes discrete choices (top-k, flags,
rebalance bands) so autodiff can flow. Nothing in that construction *proves*
the smooth gradient predicts improvement of the realized objective — the
relaxation could point uphill on the surrogate and nowhere on the hard book.
This lane measures it:

- ``hard_grad``: central finite differences of the hardened book's objective.
- ``cosine``: cos(g_smooth, g_hard) per sampled parameter point.
- ``sign_share``: share of coordinates where both gradients are nonzero and
  agree in sign.
- ``phantom_share``: coordinates where the smooth gradient is nonzero but the
  hard gradient is zero — gradient the surrogate invents across flat regions.
- ``step_helps``: does one projected ascent step along g_smooth raise the
  hard objective?

A surrogate with high phantom share or negative mean cosine is evidence the
optimizer's answers don't transfer — worth knowing before trusting the
radius certifications built on top.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.diffbacktest.jax_core import available, objective_gradients
from quant_fund.diffbacktest.numpy_core import (
    objective_value,
    simulate,
    synthetic_prices,
    warmup_start,
)
from quant_fund.diffbacktest.spec import (
    BOXES,
    GRAD_BETA,
    LIVE_PNL_CLAIM,
    RESEARCH_ONLY,
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

_EPS = 1e-10


def _hard_objective(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    objective: str,
) -> float:
    """Hardened-book objective; ``nan`` where the point is infeasible."""
    try:
        book = harden(params)
        sim = simulate(prices, strategy, book)
    except ValueError:
        return float("nan")
    return objective_value(
        sim.net,
        objective,
        periods_per_year=book.periods_per_year,
        score_start=warmup_start(strategy, book),
    )


def _hard_fd_gradient(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    names: tuple[str, ...],
    objective: str,
    *,
    rel_step: float,
) -> np.ndarray:
    """Central finite differences over the active differentiable knobs."""
    theta = pack(params, names)
    grad = np.zeros(len(names))
    for j, name in enumerate(names):
        lo, hi = BOXES[name]
        h = rel_step * (hi - lo)
        up_theta = min(theta[j] + h, hi)
        dn_theta = max(theta[j] - h, lo)
        eye = np.eye(len(names))[j]
        up = _hard_objective(
            prices,
            strategy,
            unpack(theta + (up_theta - theta[j]) * eye, names, params),
            objective,
        )
        dn = _hard_objective(
            prices,
            strategy,
            unpack(theta + (dn_theta - theta[j]) * eye, names, params),
            objective,
        )
        # infeasible side carries no slope information — fall back one-sided
        if np.isnan(up) and np.isnan(dn):
            grad[j] = 0.0
        elif np.isnan(up) and dn_theta < theta[j]:
            base_val = _hard_objective(prices, strategy, params, objective)
            grad[j] = (base_val - dn) / (theta[j] - dn_theta) if not np.isnan(base_val) else 0.0
        elif np.isnan(dn) and up_theta > theta[j]:
            base_val = _hard_objective(prices, strategy, params, objective)
            grad[j] = (up - base_val) / (up_theta - theta[j]) if not np.isnan(base_val) else 0.0
        else:
            span = up_theta - dn_theta
            grad[j] = (up - dn) / span if span > 0 else 0.0
    return grad


def grad_fidelity(
    prices: np.ndarray,
    strategy: str,
    *,
    objective: str = "sharpe",
    n_samples: int = 24,
    seed: int = 0,
    rel_step: float = 1e-3,
    beta: float = GRAD_BETA,
    step: float = 0.05,
) -> dict[str, Any]:
    """Sample parameter points; compare smooth vs hard gradient geometry."""
    if not available():
        return {"status": "jax_unavailable", "strategy": strategy}
    rng = np.random.default_rng(seed)
    names = active_parameters(strategy)
    lo = np.array([BOXES[n][0] for n in names])
    hi = np.array([BOXES[n][1] for n in names])

    cosines: list[float] = []
    sign_shares: list[float] = []
    phantom_shares: list[float] = []
    step_deltas: list[float] = []
    base = StrategyParams()
    for _ in range(n_samples):
        # boxes admit infeasible combos (e.g. lookback <= skip) — reject-resample
        params = None
        for _try in range(64):
            cand = project_box(unpack(rng.uniform(lo, hi), names, base), names)
            try:
                validate_params(strategy, cand)
            except ValueError:
                continue
            params = cand
            break
        if params is None:
            continue
        theta = pack(params, names)

        smooth = objective_gradients(prices, strategy, params, objective, mode="smooth", beta=beta)
        g_s = np.asarray(smooth.parameter_gradient, dtype=np.float64)
        g_h = _hard_fd_gradient(prices, strategy, params, names, objective, rel_step=rel_step)

        s_mag, h_mag = np.abs(g_s) > _EPS, np.abs(g_h) > _EPS
        both = s_mag & h_mag
        if both.any():
            sign_shares.append(float((np.sign(g_s[both]) == np.sign(g_h[both])).mean()))
        if s_mag.any():
            phantom_shares.append(float((~h_mag & s_mag).sum() / s_mag.sum()))
        denom = float(np.linalg.norm(g_s) * np.linalg.norm(g_h))
        cosines.append(float(g_s @ g_h / denom) if denom > _EPS else 0.0)

        norm = float(np.linalg.norm(g_s))
        if norm > _EPS:
            theta_up = theta + step * (hi - lo) * g_s / norm
            up_params = project_box(unpack(theta_up, names, params), names)
            up_val = _hard_objective(prices, strategy, up_params, objective)
            base_val = _hard_objective(prices, strategy, params, objective)
            if np.isfinite(up_val) and np.isfinite(base_val):
                step_deltas.append(float(up_val - base_val))

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    return {
        "status": "ok",
        "strategy": strategy,
        "objective": objective,
        "n_samples": n_samples,
        "mean_cosine": _mean(cosines),
        "negative_cosine_share": float(np.mean([c < 0 for c in cosines])),
        "mean_sign_share": _mean(sign_shares),
        "mean_phantom_share": _mean(phantom_shares),
        "step_helps_share": float(np.mean([d > 0 for d in step_deltas])) if step_deltas else None,
        "cosines": cosines,
    }


def grad_fidelity_bench(
    *, strategy: str = "tsmom", n_samples: int = 24, seed: int = 0
) -> dict[str, Any]:
    """Sealed ``grad_fidelity.v1`` receipt on the synthetic price panel."""
    prices = synthetic_prices(300, 6, seed)
    result = grad_fidelity(prices, strategy, n_samples=n_samples, seed=seed)
    result.pop("cosines")  # bound the receipt; moments carry the verdict
    out: dict[str, Any] = {
        "kind": "grad_fidelity",
        "schema": "grad_fidelity.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": RESEARCH_ONLY,
        "live_pnl_claim": LIVE_PNL_CLAIM,
        "claim": {
            "invariant": "smooth-surrogate gradients should predict hard-book improvement",
            "prices": "synthetic_prices(300, 6)",
            "verdict": result["status"],
        },
        "interpretation": result,
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


__all__ = ["grad_fidelity", "grad_fidelity_bench"]
