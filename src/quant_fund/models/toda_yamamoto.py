"""Toda-Yamamoto augmented-lag Granger non-causality testing.

References
----------
- Toda, H.Y. & Yamamoto, T. (1995). "Statistical Inference in
  Vector Autoregressions with Possibly Integrated Processes."
  *Journal of Econometrics* 66(1-2), 225-250.
- Dolado, J.J. & Lutkepohl, H. (1996). "Making Wald Tests Work
  for Cointegrated VAR Systems." *Econometric Reviews* 15(4),
  369-386.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The TY procedure fits a VAR in *levels* with ``p + d`` lags, where
``p`` is the chosen lag order (AIC/BIC) and ``d`` is the maximal
suspected integration order (typically 1). The Wald statistic for
Granger non-causality restricts only the first ``p`` lags of the
candidate cause in the target equation; the extra ``d`` lags are
left unrestricted. Under the TY theorem the statistic is
asymptotically chi-square with ``p`` degrees of freedom regardless
of unit roots/cointegration — this is the point of the procedure
and is what the synth must exhibit: on a bivariate system with a
planted one-directional link and I(1) driving series, the test must
reject the true direction and retain the false one at roughly the
nominal size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _design(y: FloatArray, p: int, d: int) -> tuple[FloatArray, FloatArray]:
    """Levels-VAR design: rows t = p+d .. T-1, columns [const, lag1..lag(p+d)]."""
    yy = np.asarray(y, dtype=np.float64)
    t_obs, k = yy.shape
    n_lags = p + d
    n_rows = t_obs - n_lags
    if n_rows < k * n_lags + k + 4:
        raise ValueError("series too short for (p + d) lags")
    x = np.empty((n_rows, 1 + k * n_lags))
    x[:, 0] = 1.0
    for lag in range(1, n_lags + 1):
        x[:, 1 + (lag - 1) * k : 1 + lag * k] = yy[n_lags - lag : t_obs - lag]
    return yy[n_lags:], x


def var_select_order(y: FloatArray, pmax: int = 8) -> int:
    """Select VAR lag order by AIC over the levels design (d excluded)."""
    yy = np.asarray(y, dtype=np.float64)
    best_ic, best_p = np.inf, 1
    for p in range(1, pmax + 1):
        yt, x = _design(yy, p, 0)
        beta, *_ = np.linalg.lstsq(x, yt, rcond=None)
        resid = yt - x @ beta
        s = resid.T @ resid / yt.shape[0]
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0:
            continue
        n_params = x.shape[1] * yy.shape[1]
        aic = float(logdet) + 2.0 * n_params / yt.shape[0]
        if aic < best_ic:
            best_ic, best_p = aic, p
    return best_p


def granger_mwald(
    y: FloatArray,
    cause: int,
    target: int,
    p: int,
    d: int = 1,
) -> dict[str, float]:
    """TY MWALD test: does ``cause`` Granger-cause ``target``?

    Fits the levels VAR(p + d), tests the joint restriction that
    the first ``p`` lags of series ``cause`` are zero in the
    equation for series ``target``; the last ``d`` lags stay
    unrestricted. Returns the Wald statistic, df (p), p-value, and
    the estimated residual variance.
    """
    yy = np.asarray(y, dtype=np.float64)
    if yy.ndim != 2 or yy.shape[1] < 2:
        raise ValueError("need a (T, K>=2) matrix")
    if not np.all(np.isfinite(yy)):
        raise ValueError("non-finite data")
    k = yy.shape[1]
    if not (0 <= cause < k) or not (0 <= target < k):
        raise ValueError("bad column index")
    if cause == target:
        raise ValueError("cause must differ from target")
    if p < 1 or d < 0:
        raise ValueError("need p>=1, d>=0")
    yt, x = _design(yy, p, d)
    beta, *_ = np.linalg.lstsq(x, yt, rcond=None)
    resid = yt[:, target] - x @ beta[:, target]
    dof_resid = yt.shape[0] - x.shape[1]
    if dof_resid <= 0:
        raise ValueError("insufficient degrees of freedom")
    s2 = float(resid @ resid / dof_resid)
    if s2 <= 0.0:
        raise ValueError("degenerate fit")
    xtx_inv = np.linalg.pinv(x.T @ x)
    # restricted params: lags 1..p of `cause` in `target` equation
    cols = [1 + (lag - 1) * k + cause for lag in range(1, p + 1)]
    r_b = beta[cols, target]
    r_xtx = xtx_inv[np.ix_(cols, cols)]
    stat = float(r_b @ np.linalg.pinv(r_xtx) @ r_b / s2)
    stat = max(stat, 0.0)
    p_val = float(_stats.chi2.sf(stat, df=p))
    return {
        "wald": stat,
        "df": float(p),
        "p_value": p_val,
        "sigma2": s2,
        "n_obs": float(yt.shape[0]),
    }


def synth_ty(
    seed: int = 20261231 + 298,
    t: int = 400,
    g: float = 0.5,
) -> FloatArray:
    """SYNTHETIC bivariate system: x (I(1)) Granger-causes y only."""
    rng = np.random.default_rng(seed)
    ex = rng.standard_normal(t)
    ey = rng.standard_normal(t) * 0.6
    x = np.empty(t)
    y = np.empty(t)
    x[0], y[0] = 0.0, 0.0
    for i in range(1, t):
        x[i] = x[i - 1] + ex[i]  # random walk driver
        y[i] = 0.4 * y[i - 1] + g * x[i - 1] + ey[i]
    return np.column_stack([x, y])


def bench_toda_yamamoto(seed: int = 20261231 + 298) -> dict[str, float]:
    """Wave-51 self-check: rejects true direction, keeps false."""
    y = synth_ty(seed=seed)
    p = var_select_order(y, pmax=6)
    fwd = granger_mwald(y, cause=0, target=1, p=p, d=1)
    rev = granger_mwald(y, cause=1, target=0, p=p, d=1)
    ok = fwd["p_value"] < 0.01 and rev["p_value"] > 0.01
    return {
        "synthetic_p_lag": float(p),
        "synthetic_wald_fwd": fwd["wald"],
        "synthetic_pval_fwd": fwd["p_value"],
        "synthetic_wald_rev": rev["wald"],
        "synthetic_pval_rev": rev["p_value"],
        "synthetic_score": float(ok),
    }
