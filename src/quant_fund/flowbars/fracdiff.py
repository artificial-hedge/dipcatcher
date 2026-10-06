"""Fixed-window fractional differentiation (López de Prado, AFML ch. 5).

Integer differencing (d = 1) removes a unit root but throws away a lot of
memory; fractional differentiation with 0 < d < 1 stationarises a series
while preserving far more of its signal. This module provides:

- ``frac_diff_weights`` — the (1 − B)^d weight sequence;
- ``frac_diff_apply`` — causal fixed-window application (FFD);
- ``adf_pvalue`` / ``min_stationary_d`` — the ADF-based grid scan for the
  smallest d that stationarises the series. The ADF test has limited power
  against long-memory alternatives, so the scan's recovered d should be read
  as approximate near fractional orders;
- ``synth_frac_integrated`` — an exact FI(d) generator for round-trip tests.

Honesty: the ADF scan is a stationarity diagnostic, not a predictive claim.
The generator is labeled synthetic.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 5 — fixed-width window fractional differentiation.
- Hosking, J. R. M. (1981). Fractional differencing — the weight recurrence
  and FI(d) processes.
- Dickey, D. A., Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root — the ADF test.

Composition: numpy + statsmodels (ADF, already locked); deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller

FloatArray = NDArray[np.float64]


def frac_diff_weights(d: float, size: int) -> FloatArray:
    """Weights w_k of (1 − B)^d: w_0 = 1, w_k = w_{k−1}·(k−1−d)/k.

    For integer d these collapse to the finite-difference weights (d = 1
    gives [1, −1, 0, …]). For 0 < d < 1 all higher-order weights are
    negative and decay as k^{−(1+d)}.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    w = np.zeros(size, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def frac_integrate_weights(d: float, size: int) -> FloatArray:
    """Weights g_k of (1 − B)^{−d}: g_0 = 1, g_k = g_{k−1}·(k−1+d)/k."""
    if size < 1:
        raise ValueError("size must be >= 1")
    g = np.zeros(size, dtype=np.float64)
    g[0] = 1.0
    for k in range(1, size):
        g[k] = g[k - 1] * (k - 1 + d) / k
    return g


def frac_diff_apply(x: FloatArray, d: float, window: int | None = None) -> FloatArray:
    """Causal fixed-window fractional difference of ``x``.

    y_t = Σ_{k=0}^{min(t, window−1)} w_k · x_{t−k}. With ``window=None`` the
    full history is used (exact, O(n²)); with a finite window the series
    keeps its length, unlike integer diff which loses observations.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    n = len(x)
    w_full = frac_diff_weights(d, n if window is None else min(window, n))
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        k_max = min(t, len(w_full) - 1)
        out[t] = float(np.dot(w_full[: k_max + 1], x[t - k_max : t + 1][::-1]))
    return out


def adf_pvalue(x: FloatArray) -> float:
    """ADF test p-value (constant series fail closed to p = 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 10:
        raise ValueError("x must have at least 10 observations")
    if float(np.std(x, ddof=1)) <= 0:
        return 1.0
    stat = adfuller(x, regression="c", autolag="AIC")
    return float(stat[1])


def min_stationary_d(
    x: FloatArray,
    d_grid: FloatArray | None = None,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Smallest d on the grid whose FFD(d) is stationary by ADF.

    Returns ``{"d": d_star, "pvalue": p, "alpha": alpha}``; when no grid
    point passes, returns the largest d with its p-value and lets the caller
    decide.
    """
    x = np.asarray(x, dtype=np.float64)
    if d_grid is None:
        d_grid = np.linspace(0.0, 1.0, 21)
    grid = np.asarray(d_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("d_grid must be a non-empty one-dimensional array")
    for d in grid:
        y = frac_diff_apply(x, float(d))
        p = adf_pvalue(y)
        if p < alpha:
            return {"d": float(d), "pvalue": p, "alpha": float(alpha)}
    return {
        "d": float(grid[-1]),
        "pvalue": adf_pvalue(frac_diff_apply(x, float(grid[-1]))),
        "alpha": float(alpha),
    }


def synth_frac_integrated(
    n: int = 2000,
    *,
    d: float = 0.4,
    seed: int = 0,
) -> FloatArray:
    """Exact FI(d) series: y = (1 − B)^{−d} ε with ε ~ N(0, 1).

    Labeled synthetic. The series is fractionally integrated of order d, so
    FFD with the same d recovers (approximately) white noise — the property
    ``min_stationary_d`` tests exploit.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0.0 <= d < 0.5:
        raise ValueError("d must be in [0, 0.5) for a stationary FI(d) series")
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    g = frac_integrate_weights(d, n)
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        k_max = min(t, len(g) - 1)
        out[t] = float(np.dot(g[: k_max + 1], eps[t - k_max : t + 1][::-1]))
    return out
