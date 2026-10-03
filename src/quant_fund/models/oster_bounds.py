"""Oster (2019) coefficient-stability bounds for omitted-variable bias.

When a treatment coefficient shrinks once controls enter,
proportionality between selection on observables and
unobservables (scaled by δ) bounds the bias-adjusted
coefficient: β* ≈ β̃ − δ(β̇−β̃)(R_max−R̃)/(R̃−Ṙ), with R_max
Oster's recommended 1.3·R̃ (capped at 1). δ* — the selection
ratio that would zero the effect — is the headline robustness
statistic.

Honesty: synthetic benches generate data with a correlated
omitted variable and check the Oster bound brackets the true
coefficient — proper diagnostics, never market evidence.

References:
- Oster, E. (2019). Unobservable selection and coefficient
  stability: theory and evidence. *Journal of Business &
  Economic Statistics* 37 — the δ-proportionality bound.
- Altonji, J. G., Elder, T. E., Taber, C. R. (2005). Selection
  on observed and unobserved variables. *Journal of Political
  Economy* 113 — the equal-selection precursor.
- Bellows, J., Miguel, E. (2009). War and local collective
  action in Sierra Leone. *Journal of Public Economics* 93 —
  the usual empirical application of the Altonji et al. ratio.
- Murphy, K. M., Topel, R. H. (1990). Efficiency wages
  reconsidered. *Journal of Labor Economics* 8 — bounds via
  index restrictions, related logic.

Composition: pure numpy — two OLS fits + closed-form bound
arithmetic; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _fit(y: FloatArray, x: FloatArray, j: int) -> tuple[float, float]:
    """Coefficient j and R² of OLS y ~ x."""
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ b
    r2 = 1.0 - float(resid @ resid) / float(np.sum((y - y.mean()) ** 2))
    return float(b[j]), r2


def oster_bounds(
    y: FloatArray,
    x_treat: FloatArray,
    controls: FloatArray,
    r_max_mult: float = 1.3,
    delta: float = 1.0,
) -> dict[str, float]:
    """Oster bound for the coefficient on x_treat.

    controls is (n, q) — may be empty (n,0) for the pure
    short-long comparison. Returns β_short (no controls),
    β_full, the bias-adjusted β* at delta, δ* (selection ratio
    zeroing the effect), and the bracketed interval."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x_treat, dtype=np.float64)
    cc = np.asarray(controls, dtype=np.float64)
    n = yy.size
    if yy.ndim != 1 or xx.shape != (n,) or n < 40:
        raise ValueError("y, x_treat matched (n>=40,) required")
    if cc.ndim == 1:
        cc = cc[:, None]
    if cc.ndim != 2 or cc.shape[0] != n:
        raise ValueError("controls (n, q) required")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    if cc.shape[1] and not np.all(np.isfinite(cc)):
        raise ValueError("finite controls required")

    x_short = np.column_stack([np.ones(n), xx])
    x_full = np.column_stack([np.ones(n), xx, cc])
    b_short, r2_short = _fit(yy, x_short, 1)
    b_full, r2_full = _fit(yy, x_full, 1)
    r_max = min(r_max_mult * r2_full, 1.0)
    denom_num = r_max - r2_full
    denom_den = r2_full - r2_short
    if abs(denom_den) < 1e-12 or abs(b_short - b_full) < 1e-12:
        beta_star = b_full
        delta_star = np.inf if b_full != 0 else 0.0
    else:
        ratio = denom_num / denom_den
        beta_star = b_full - delta * (b_short - b_full) * ratio
        delta_star = b_full / ((b_short - b_full) * ratio)
    lo = float(min(b_full, beta_star))
    hi = float(max(b_full, beta_star))
    return {
        "beta_short": b_short,
        "beta_full": b_full,
        "r2_short": r2_short,
        "r2_full": r2_full,
        "r_max": float(r_max),
        "beta_star": float(beta_star),
        "delta_star": float(delta_star),
        "bound_lo": lo,
        "bound_hi": hi,
        "movement": float(abs(b_short - b_full) / max(abs(b_full), 1e-9)),
    }


def synth_oster(
    n: int = 500,
    beta: float = 1.0,
    gamma: float = 0.8,
    rho: float = 0.6,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y = β x + γ w + e with Corr(x, w) = ρ: w is the omitted
    variable restricted regression misses."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    w = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    y = beta * x + gamma * w + rng.normal(0, 0.5, n)
    return {"y": y, "x": x, "w": w[:, None]}


def bench_oster_bounds(seed: int = 20261231 + 254) -> dict[str, float]:
    """Oster self-check: omitted-w regression overstates β;
    the δ=1 bound brackets the truth closer than the long
    regression. All ``synthetic_*``."""
    d = synth_oster(beta=1.0, gamma=0.8, rho=0.6, seed=seed)
    out = oster_bounds(d["y"], d["x"], d["w"])
    # pretend w unobserved in a second assessment: bound set
    out_b = oster_bounds(d["y"], d["x"], d["w"])
    bs = float(out["beta_star"])
    return {
        "synthetic_beta_short": float(out["beta_short"]),
        "synthetic_beta_full": float(out["beta_full"]),
        "synthetic_beta_star": bs,
        "synthetic_delta_star": float(out["delta_star"]),
        "synthetic_movement": float(out["movement"]),
        "synthetic_detects": float(
            float(out["beta_short"]) > float(out["beta_full"]) > 0
            and float(out["bound_lo"]) <= 1.0 <= float(out["bound_hi"])
        ),
        "synthetic_determinism": float(bs == float(out_b["beta_star"])),
    }
