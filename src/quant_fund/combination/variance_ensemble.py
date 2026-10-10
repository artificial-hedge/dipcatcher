"""Variance-forecast members, QLIKE evaluation, and Diebold–Mariano tests.

Builds a small family of variance-forecast members from a return series,
evaluates them with QLIKE, and compares pairs with the Diebold–Mariano test:

- ``ewma_variance`` — RiskMetrics-style EWMA with forgetting λ;
- ``rolling_mad_variance`` — rolling median-absolute-deviation scale, a
  jump-robust member;
- ``long_run_variance`` — expanding/rolling flat member as the anchor;
- ``qlike_losses`` — per-observation QLIKE losses for each member;
- ``dm_test`` — Diebold–Marino (1995) test for equal predictive accuracy on
  a generic loss differential (QLIKE by default), HAR-adjusted variance for
  the mean of the loss difference.

Honesty: members are variance *proxies*; QLIKE is robust to noisy proxies
(Patton 2011), and all evaluation is on supplied data.

References:
- Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy —
  the DM test.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies — QLIKE robustness.
- Hansen, P. R., Lunde, A. (2006). Realized variance and market
  microstructure noise — variance proxy caveats.

Composition: numpy + scipy (locked); deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _returns(x: FloatArray) -> FloatArray:
    r = np.asarray(x, dtype=np.float64)
    if r.ndim != 1 or len(r) < 10:
        raise ValueError("x must be a one-dimensional array with >= 10 points")
    return r


def ewma_variance(returns: FloatArray, lam: float = 0.94) -> FloatArray:
    """EWMA variance forecast (one-step-ahead) aligned with the returns."""
    r = _returns(returns)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    var = float(np.var(r[:10]))
    for t in range(10, n):
        out[t] = var
        var = lam * var + (1.0 - lam) * r[t] ** 2
    return out


def rolling_mad_variance(returns: FloatArray, window: int = 20) -> FloatArray:
    """Rolling median-absolute-deviation variance proxy, jump-robust."""
    r = _returns(returns)
    n = len(r)
    if window < 5:
        raise ValueError("window must be >= 5")
    out = np.full(n, np.nan, dtype=np.float64)
    scale = 1.4826
    for t in range(window, n):
        w = r[t - window : t]
        mad = float(np.median(np.abs(w - np.median(w))))
        out[t] = (scale * mad) ** 2
    return out


def long_run_variance(returns: FloatArray, window: int | None = None) -> FloatArray:
    """Flat variance member: expanding or rolling mean of r²."""
    r = _returns(returns)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(1, n):
        lo = 0 if window is None else max(0, t - window)
        out[t] = float(np.mean(r[lo:t] ** 2))
    return out


def qlike_losses(realized: FloatArray, forecasts: FloatArray) -> FloatArray:
    """Per-observation QLIKE loss log(v) + y²/v with fail-closed masking."""
    y = np.asarray(realized, dtype=np.float64)
    v = np.asarray(forecasts, dtype=np.float64)
    if y.shape != v.shape:
        raise ValueError("realized and forecasts must have equal shapes")
    valid = np.isfinite(y) & np.isfinite(v) & (v > 0)
    out = np.full(len(y), np.nan, dtype=np.float64)
    out[valid] = np.log(v[valid]) + y[valid] ** 2 / v[valid]
    return out


def dm_test(
    losses_a: FloatArray,
    losses_b: FloatArray,
    *,
    h: int = 1,
) -> dict[str, float]:
    """Diebold–Mariano test on the loss differential d = a − b.

    H0: E[d] = 0 (equal accuracy). The long-run variance of d̄ uses HAC
    with h−1 lags; the statistic is asymptotically standard normal.
    """
    da = np.asarray(losses_a, dtype=np.float64)
    db = np.asarray(losses_b, dtype=np.float64)
    if da.shape != db.shape or da.ndim != 1:
        raise ValueError("losses must be one-dimensional arrays of equal shape")
    valid = np.isfinite(da) & np.isfinite(db)
    d = (da - db)[valid]
    n = len(d)
    if n < 20:
        raise ValueError("need at least 20 paired losses")
    d_bar = float(np.mean(d))
    gamma0 = float(np.mean((d - d_bar) ** 2))
    lr_var = gamma0
    for lag in range(1, h):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        lr_var += 2.0 * (1.0 - lag / h) * cov
    if lr_var <= 0:
        return {"stat": 0.0, "p": 1.0, "diff": d_bar}
    stat = d_bar / np.sqrt(lr_var / n)
    from scipy import stats as sps

    p = float(2.0 * (1.0 - sps.norm.cdf(abs(stat))))
    return {"stat": float(stat), "p": p, "diff": d_bar}
