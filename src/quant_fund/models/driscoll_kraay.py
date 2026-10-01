"""Driscoll-Kraay standard errors for panels.

DK robustness treats each period's cross-sectional aggregate
score h_t = Σ_i x_it e_it as a time series and applies a
HAC correction over t — robust to arbitrary cross-sectional
dependence, spatial correlation, and serial correlation,
needing only T moderately large.

Honesty: synthetic benches generate panels with a common
factor (cross-sectional dependence) and aggregate scores pick up the serial structure — DK bandwidth inflation over time-clustered
relative to cluster-iid SEs — proper diagnostics, never
market evidence.

References:
- Driscoll, J. C., Kraay, A. C. (1998). Consistent covariance
  matrix estimation with spatially dependent panel data.
  *Review of Economics and Statistics* 80 — the estimator.
- Hoechle, D. (2007). Robust standard errors for panel
  regressions with cross-sectional dependence. *Stata
  Journal* 7 — the small-sample T/(T−1) adjustment and
  bandwidth guidance used here.
- Vogelsang, T. J. (2012). Heteroskedasticity,
  autocorrelation, and spatial correlation robust inference
  in linear panel models. *Journal of Econometrics* 167 —
  fixed-b interpretation.
- Conley, T. G. (1999). GMM estimation with cross sectional
  dependence. *Journal of Econometrics* 92 — the limit
  DK nests.

Composition: pure numpy — aggregated-score Bartlett HAC;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _bartlett(u: FloatArray) -> FloatArray:
    return np.maximum(0.0, 1.0 - np.abs(u))


def driscoll_kraay_vcov(
    x: FloatArray,
    resid: FloatArray,
    t_idx: FloatArray,
    m: float = 2.0,
) -> FloatArray:
    """DK variance for OLS coefs on stacked panel rows.

    x is (N,k) stacked rows (period-major or any order);
    t_idx is (N,) integer period index per row; resid (N,).
    Aggregates scores by period then Bartlett-HAC over time."""
    xx = np.asarray(x, dtype=np.float64)
    e = np.asarray(resid, dtype=np.float64)
    ti = np.asarray(t_idx, dtype=np.float64)
    n = xx.shape[0]
    if xx.ndim != 2 or e.shape != (n,) or ti.shape != (n,) or n < 60:
        raise ValueError("x (N,k), e, t_idx matched, N>=60 required")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(e)):
        raise ValueError("finite inputs required")
    if m < 1:
        raise ValueError("m >= 1 required")
    # aggregate scores per period
    periods = np.unique(ti)
    t_n = periods.size
    h = np.zeros((t_n, xx.shape[1]))
    for j, tp in enumerate(periods):
        sel = ti == tp
        h[j] = (xx[sel] * e[sel, None]).sum(axis=0)
    # Bartlett HAC over periods
    i = np.arange(t_n)
    w = _bartlett((i[:, None] - i[None, :]) / m)
    meat = h.T @ w @ h
    bread = np.linalg.inv(xx.T @ xx)
    # small-sample adjustment (Hoechle 2007): T/(T-1) on meat
    v = bread @ meat @ bread * (t_n / (t_n - 1))
    return np.asarray(v, dtype=np.float64)


def synth_dk_panel(
    n_per: int = 40,
    t: int = 60,
    beta: float = 1.0,
    factor_sd: float = 1.5,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Panel: y_it = β x_it + f_t + u_it — common factor f_t
    creates cross-sectional dependence."""
    rng = np.random.default_rng(seed)
    f = np.zeros(t)
    if factor_sd > 0:
        f[0] = rng.normal(0, factor_sd)
        for i in range(1, t):
            f[i] = 0.95 * f[i - 1] + rng.normal(0, factor_sd)
    x = np.zeros((t, n_per))
    x[0] = rng.normal(0, 1, n_per)
    for i in range(1, t):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 1, n_per)
    u = rng.normal(0, 1, (t, n_per))
    y = beta * x + f[:, None] + u
    ti = np.repeat(np.arange(t), n_per)
    return {
        "y": y.ravel(),
        "x": x.ravel()[:, None],
        "t_idx": ti.astype(np.float64),
    }


def bench_driscoll_kraay(seed: int = 20261231 + 258) -> dict[str, float]:
    """DK self-check: common-factor panel inflates DK SE over
    the iid SE several-fold; no factor → near parity.
    All ``synthetic_*``."""
    d = synth_dk_panel(factor_sd=1.5, seed=seed)
    x = np.asarray(d["x"])
    xd = np.column_stack([np.ones(x.shape[0]), x])
    b, *_ = np.linalg.lstsq(xd, np.asarray(d["y"]), rcond=None)
    e = np.asarray(d["y"]) - xd @ b
    v = driscoll_kraay_vcov(xd, e, np.asarray(d["t_idx"]), m=4.0)
    se_dk = float(np.sqrt(v[1, 1]))
    # m=1 ≈ pure time-clustered (no serial-correlation capture)
    v1 = driscoll_kraay_vcov(xd, e, np.asarray(d["t_idx"]), m=1.0)
    se_w = float(np.sqrt(v1[1, 1]))
    d0 = synth_dk_panel(factor_sd=0.0, seed=seed + 1)
    x0 = np.column_stack([np.ones(d0["x"].shape[0]), np.asarray(d0["x"])])
    b0, *_ = np.linalg.lstsq(x0, np.asarray(d0["y"]), rcond=None)
    e0 = np.asarray(d0["y"]) - x0 @ b0
    v0 = driscoll_kraay_vcov(x0, e0, np.asarray(d0["t_idx"]), m=4.0)
    v10 = driscoll_kraay_vcov(x0, e0, np.asarray(d0["t_idx"]), m=1.0)
    ratio_null = float(np.sqrt(v0[1, 1]) / np.sqrt(v10[1, 1]))
    ratio = se_dk / se_w
    v_b = driscoll_kraay_vcov(xd, e, np.asarray(d["t_idx"]), m=4.0)
    return {
        "synthetic_se_dk": se_dk,
        "synthetic_se_clustered": se_w,
        "synthetic_inflation": ratio,
        "synthetic_inflation_null": ratio_null,
        "synthetic_detects": float(ratio > 1.3 and ratio_null < ratio),
        "synthetic_determinism": float(float(np.sqrt(v_b[1, 1])) == se_dk),
    }
