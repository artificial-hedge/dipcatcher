"""Pesaran (2006) Common Correlated Effects estimator.

When panel errors share unobserved factors, pooled/FE slopes
are biased; CCE augments each unit's regression with the
cross-sectional means of y and x as factor proxies. CCEMG
takes the mean of per-unit slopes; CCEP pools.

Honesty: synthetic benches generate a one-factor panel and
check CCEMG recovers β where OLS-FE is biased — proper
diagnostics, never market evidence.

References:
- Pesaran, M. H. (2006). Estimation and inference in large
  heterogeneous panels with a multifactor error structure.
  *Econometrica* 74 — CCEMG/CCEP and rank conditions.
- Kapetanios, G., Pesaran, M. H., Yamagata, T. (2011).
  Panels with non-stationary multifactor error structures.
  *Journal of Econometrics* 160 — robustness.
- Pesaran, M. H., Tosetti, E. (2011). Large panels with
  common factors and spatial correlation. *Journal of
  Econometrics* 161 — extensions.
- Chudik, A., Pesaran, M. H. (2015). Common correlated
  effects estimation of heterogeneous dynamic panel data
  models with weakly exogenous regressors. *Journal of
  Econometrics* 188.

Composition: pure numpy — per-unit augmented OLS + mean of
slopes; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cce_slopes(
    y: FloatArray,
    x: FloatArray,
    i_idx: FloatArray,
) -> FloatArray:
    """Per-unit CCE slope: regress y_i on [x_i, ybar_t, xbar_t]
    and keep the x coefficient. y/x are (N,) stacked rows with
    unit index i_idx; t index must be reconstructable — pass
    rows period-major OR provide the same t per group. Here we
    require (n_per × t) blocks per unit contiguous and equal
    length t."""
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    ii = np.asarray(i_idx, dtype=np.float64)
    n = yy.shape[0]
    if xx.shape != (n,) or ii.shape != (n,) or n < 60:
        raise ValueError("matched (N,) arrays, N>=60 required")
    if not np.all(np.isfinite(yy)) or not np.all(np.isfinite(xx)):
        raise ValueError("finite inputs required")
    units, first = np.unique(ii, return_index=True)
    sizes = np.diff(np.append(first, n))
    if len(np.unique(sizes)) != 1:
        raise ValueError("balanced panels only")
    t = int(sizes[0])
    # cross-sectional means at each t (period-major within units)
    ym = np.zeros(t)
    xm = np.zeros(t)
    for u in units:
        sel = ii == u
        ym += yy[sel]
        xm += xx[sel]
    ym /= units.size
    xm /= units.size
    betas = np.zeros(units.size)
    for j, u in enumerate(units):
        sel = ii == u
        xd = np.column_stack([np.ones(t), xx[sel], ym, xm])
        coef = np.linalg.lstsq(xd, yy[sel], rcond=None)[0]
        betas[j] = coef[1]
    return np.asarray(betas, dtype=np.float64)


def cce_mean_group(y: FloatArray, x: FloatArray, i_idx: FloatArray) -> float:
    """CCEMG: mean of per-unit CCE slopes."""
    return float(np.mean(cce_slopes(y, x, i_idx)))


def synth_cce_panel(
    n_per: int = 30,
    t: int = 50,
    beta: float = 1.0,
    gamma_x: float = 1.2,
    gamma_y: float = 1.0,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """y_it = β x_it + γ_y f_t + u; x_it = γ_x f_t + v —
    common factor contaminates OLS."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, t)
    x = gamma_x * f[:, None] + rng.normal(0, 1, (t, n_per))
    u = rng.normal(0, 1, (t, n_per))
    y = beta * x + gamma_y * f[:, None] + u
    ii = np.tile(np.arange(n_per), (t, 1)).T  # unit-major
    return {
        "y": y.T.ravel(),
        "x": x.T.ravel(),
        "i_idx": ii.ravel().astype(np.float64),
    }


def bench_pesaran_cce(seed: int = 20261231 + 259) -> dict[str, float]:
    """CCE self-check: OLS biased toward β+γ_y/γ_x ≈ 1.83;
    CCEMG recovers ≈β=1.0. All ``synthetic_*``."""
    d = synth_cce_panel(beta=1.0, seed=seed)
    beta_mg = cce_mean_group(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["i_idx"]))
    # naive pooled OLS on demeaned
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    b_ols = float(np.polyfit(x, y, 1)[0])
    b2 = cce_mean_group(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["i_idx"]))
    return {
        "synthetic_beta_ccemg": beta_mg,
        "synthetic_beta_ols": b_ols,
        "synthetic_bias_ols": abs(b_ols - 1.0),
        "synthetic_bias_ccemg": abs(beta_mg - 1.0),
        "synthetic_detects": float(abs(beta_mg - 1.0) < 0.15 and abs(b_ols - 1.0) > 0.3),
        "synthetic_determinism": float(b2 == beta_mg),
    }
