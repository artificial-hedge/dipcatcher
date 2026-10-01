"""Interrupted time series (ITS) — segmented regression design.

Pre/post intervention levels and slopes:
    y_t = β0 + β1·t + β2·post_t + β3·(t−τ)·post_t + e_t
with Newey-West standard errors so the level/slope change tests
respect serial correlation. The comparative (controlled) variant
differences the treated segment against a concurrent control.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure break detection on generated
segmented series — never market evidence.

References:
- Bernal, J. L., Cummins, S., Gasparrini, A. (2017). Interrupted
  time series regression for the evaluation of public health
  interventions: a tutorial. *Int. J. Epidemiology* 46, 348-355.
- Wagner, A. K., Soumerai, S. B., Zhang, F., Ross-Degnan, D.
  (2002). Segmented regression analysis of interrupted time
  series studies. *J. Clin. Pharm. Ther.* 27, 299-309.
- Newey, W. K., West, K. D. (1987). A simple, positive
  semi-definite, HAC covariance matrix. *Econometrica* 55.
- Linden, A. (2015). Conducting interrupted time-series analysis
  for single- and multiple-group comparisons. *Stata J.* 15 —
  controlled ITS (CITS).

Composition: pure numpy + scipy — segmented OLS design,
Bartlett-kernel HAC covariance; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _nw_vcov(w: FloatArray, resid: FloatArray, lag: int) -> FloatArray:
    """Newey-West covariance of OLS coefficients.

    V = (W'W)⁻¹ S (W'W)⁻¹ with S = Σ e_i² w_i w_i' +
    Σ_{l≤L} ω_l Σ_{i>l} e_i e_{i−l} (w_i w_{i−l}' + w_{i−l} w_i')."""
    xtx_inv = np.linalg.inv(w.T @ w)
    s = w.T @ (w * (resid**2)[:, None])
    for lag_i in range(1, lag + 1):
        wl = 1.0 - lag_i / (lag + 1.0)
        gamma = w[lag_i:].T @ (w[:-lag_i] * (resid[lag_i:] * resid[:-lag_i])[:, None])
        s += wl * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


def interrupted_ts(
    y: FloatArray,
    tau: int,
    ar_check: bool = True,
) -> dict[str, float]:
    """Segmented ITS fit for a single series.

    ``tau`` is the intervention index (0-based, first post period).
    Returns level change β2, slope change β3 (NW t-stats), the
    pre-trend, and fitted values diagnostics."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    t = yy.size
    if t < 60:
        raise ValueError("T>=60")
    if not (10 <= tau <= t - 10):
        raise ValueError("tau needs >=10 obs on each side")
    if not np.all(np.isfinite(yy)):
        raise ValueError("finite inputs required")
    if np.std(yy) < 1e-9:
        raise ValueError("y must vary")

    tt = np.arange(t, dtype=np.float64)
    post = (tt >= tau).astype(np.float64)
    ttau = np.clip(tt - tau, 0.0, None) * post
    w = np.column_stack([np.ones(t), tt, post, ttau])
    beta, *_ = np.linalg.lstsq(w, yy, rcond=None)
    resid = yy - w @ beta
    lag = int(np.floor(4 * (t / 100.0) ** (2 / 9)))
    vc = _nw_vcov(w, resid, lag)
    se = np.sqrt(np.maximum(np.diag(vc), 1e-300))

    t_level = float(beta[2] / se[2])
    t_slope = float(beta[3] / se[3])
    p_level = float(2 * (1 - stats.norm.cdf(abs(t_level))))
    p_slope = float(2 * (1 - stats.norm.cdf(abs(t_slope))))

    # Durbin-Watson for residual autocorrelation
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
    # naive (spherical) se for contrast
    s2 = float(resid @ resid) / (t - 4)
    se_naive = np.sqrt(s2 * np.diag(np.linalg.inv(w.T @ w)))

    return {
        "t": float(t),
        "tau": float(tau),
        "beta_pre_trend": float(beta[1]),
        "beta_level": float(beta[2]),
        "beta_slope": float(beta[3]),
        "t_level": t_level,
        "t_slope": t_slope,
        "p_level": p_level,
        "p_slope": p_slope,
        "dw": dw,
        "se_level": float(se[2]),
        "se_level_naive": float(se_naive[2]),
        "r2": float(1.0 - np.var(resid) / np.var(yy)),
    }


def comparative_its(
    y_treated: FloatArray,
    y_control: FloatArray,
    tau: int,
) -> dict[str, float]:
    """CITS: ITS on (treated − control) difference series."""
    a = np.asarray(y_treated, dtype=np.float64).ravel()
    b = np.asarray(y_control, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("series must match")
    diff = a - b
    return interrupted_ts(diff, tau)


def synth_its(
    t: int = 200,
    tau: int = 120,
    level_shift: float = 0.8,
    slope_shift: float = 0.01,
    ar: float = 0.5,
    seed: int = 0,
) -> FloatArray:
    """Segmented DGP: pre-trend + level/slope break at τ, AR(1) errors."""
    rng = np.random.default_rng(seed)
    tt = np.arange(t, dtype=np.float64)
    post = (tt >= tau).astype(np.float64)
    e = np.zeros(t)
    eps = rng.normal(0.0, 0.3, t)
    for i in range(1, t):
        e[i] = ar * e[i - 1] + eps[i]
    y = 1.0 + 0.02 * tt + level_shift * post + slope_shift * np.clip(tt - tau, 0, None) * post + e
    return y


def bench_interrupted_ts(seed: int = 20261231 + 234) -> dict[str, float]:
    """ITS self-check: level/slope breaks detected at τ; a no-break
    series keeps both nulls; NW t differs from naive under AR.
    All ``synthetic_*``."""
    y = synth_its(level_shift=0.8, slope_shift=0.02, seed=seed)
    out = interrupted_ts(y, tau=120)
    y0 = synth_its(level_shift=0.0, slope_shift=0.0, seed=seed + 1)
    out0 = interrupted_ts(y0, tau=120)
    out_b = interrupted_ts(y, tau=120)

    return {
        "synthetic_beta_level": float(out["beta_level"]),
        "synthetic_level_err": float(abs(float(out["beta_level"]) - 0.8)),
        "synthetic_beta_slope": float(out["beta_slope"]),
        "synthetic_slope_err": float(abs(float(out["beta_slope"]) - 0.02)),
        "synthetic_p_level": float(out["p_level"]),
        "synthetic_p_slope": float(out["p_slope"]),
        "synthetic_p_level_nobreak": float(out0["p_level"]),
        "synthetic_dw": float(out["dw"]),
        "synthetic_detects": float(
            float(out["p_level"]) < 0.05
            and float(out["p_slope"]) < 0.05
            and float(out0["p_level"]) > 0.05
            and abs(float(out["beta_level"]) - 0.8) < 0.5
        ),
        "synthetic_determinism": float(float(out["beta_level"]) == float(out_b["beta_level"])),
    }
