"""Cross-sectional information coefficient (IC) time series.

For each period t the IC is the cross-sectional correlation (Pearson or
Spearman) between the signal and the realised forward return across assets.
The IC series is the raw material of every decay diagnostic in this
package; the t-statistics here use the time-series variation of the IC, with
an AR(1)-effective-sample-size correction for serial dependence.

Honesty: IC is a rank/linear correlation diagnostic on the supplied panel.
It is not a return forecast quality claim beyond the sample provided.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — IC, IR.
- Harvey, C. R., Liu, Y. (2015). Backtesting — serial-correlation adjustment
  of t-statistics (effective sample size treatment).

Composition: numpy + scipy.stats.rankdata (locked); deterministic.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]


def _row_corr(a: FloatArray, b: FloatArray) -> FloatArray:
    """Per-row Pearson correlation of two (T, N) matrices."""
    a_dm = a - a.mean(axis=1, keepdims=True)
    b_dm = b - b.mean(axis=1, keepdims=True)
    num = (a_dm * b_dm).sum(axis=1)
    den = np.sqrt((a_dm * a_dm).sum(axis=1) * (b_dm * b_dm).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return cast(FloatArray, np.asarray(out, dtype=np.float64))


def pearson_ic(pred: FloatArray, actual: FloatArray) -> FloatArray:
    """Per-period cross-sectional Pearson IC; NaN rows dropped by caller."""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if pred.shape != actual.shape or pred.ndim != 2:
        raise ValueError("pred and actual must be (T, N) matrices of equal shape")
    return _row_corr(pred, actual)


def spearman_ic(pred: FloatArray, actual: FloatArray) -> FloatArray:
    """Per-period cross-sectional Spearman IC (rank IC)."""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if pred.shape != actual.shape or pred.ndim != 2:
        raise ValueError("pred and actual must be (T, N) matrices of equal shape")
    r_pred = np.asarray(cast(FloatArray, rankdata(pred, axis=1)), dtype=np.float64)
    r_actual = np.asarray(cast(FloatArray, rankdata(actual, axis=1)), dtype=np.float64)
    return _row_corr(r_pred, r_actual)


def _valid(ic: FloatArray) -> FloatArray:
    out = np.asarray(ic, dtype=np.float64)
    return cast(FloatArray, out[np.isfinite(out)])


def ic_summary(ic: FloatArray) -> dict[str, float]:
    """Mean/std/t-stat/IR/skew/kurt of a (possibly NaN-padded) IC series."""
    x = _valid(ic)
    n = float(len(x))
    if n < 3:
        raise ValueError("need at least 3 valid IC observations")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    t = ic_tstat(x)
    z = (x - mu) / sd if sd > 0 else np.zeros_like(x)
    return {
        "n": n,
        "mean": mu,
        "std": sd,
        "tstat": t,
        "ir": float(mu / sd) if sd > 0 else float("nan"),
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def ic_tstat(ic: FloatArray) -> float:
    """Standard t-statistic of the IC mean (assumes i.i.d. ICs)."""
    x = _valid(ic)
    if len(x) < 3:
        raise ValueError("need at least 3 valid IC observations")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / sd * np.sqrt(len(x)))


def effective_sample_tstat(ic: FloatArray, max_lag: int = 5) -> dict[str, float]:
    """AR(1)-adjusted t-statistic using the effective sample size.

    The long-run variance of the IC mean is inflated by serial correlation;
    with lag-1 autocorrelation ρ the variance multiplier is (1+ρ)/(1−ρ).
    Returns the plain and adjusted t plus the estimated ρ.
    """
    x = _valid(ic)
    if len(x) < 10:
        raise ValueError("need at least 10 valid IC observations")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return {"tstat": float("nan"), "adjusted_tstat": float("nan"), "rho": float("nan")}
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    rho = min(max(rho, -0.95), 0.95)
    n = len(x)
    t_plain = mu / sd * np.sqrt(n)
    t_adj = t_plain * np.sqrt((1.0 - rho) / (1.0 + rho))
    return {"tstat": float(t_plain), "adjusted_tstat": float(t_adj), "rho": rho}
