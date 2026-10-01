"""Cleveland STL seasonal-trend decomposition via LOESS.

References
----------
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning,
  I. (1990). "STL: A Seasonal-Trend Decomposition Procedure
  Based on Loess." *Journal of Official Statistics* 6(1), 3-73.
- Cleveland, W.S. (1979). "Robust Locally Weighted Regression
  and Smoothing Scatterplots." *JASA* 74(368), 829-836.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
STL alternates two loops: the inner loop detrends the series,
smooths each cycle-subseries with LOESS, recenters the
seasonal by low-pass filtering the cycle-subseries and
subtracting, then re-extracts the trend by LOESS on the
seasonally-adjusted series; the outer loop forms bisquare
robustness weights on the residual and re-runs the inner loop
with weighted LOESS. We implement local-linear LOESS with
tricube weights ``w_i = (1 - |u_i|^3)^3`` on the ``win``
nearest neighbours — local linear (not local constant) is
load-bearing at the boundaries, where a constant smoother's
bias leaks the trend into the seasonal. The classic
``seasonal_decompose`` moving-average filter leaves ``period/2``
NAs at each end and no robustness weights; STL's cycle-
subseries smoothing avoids both defects. ``synth_stl`` plants
a quadratic trend + two-harmonic seasonal + noise; the bench
gates on residual-to-total variance collapse and high
correlation of the recovered trend with the planted one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 60) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _loess(
    x: FloatArray,
    y: FloatArray,
    win: int,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Local-linear LOESS with tricube weights."""
    n = x.size
    win = max(3, min(win, n))
    out = np.empty(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        h = np.partition(dist, win - 1)[win - 1]
        if h <= 0:
            h = np.max(dist) + 1e-12
        u = dist / h
        w = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
        if weights is not None:
            w = w * weights
        xc = x - x[i]
        s_w = float(np.sum(w))
        if s_w < 1e-12:
            out[i] = y[i]
            continue
        a12 = float(np.sum(w * xc))
        a22 = float(np.sum(w * xc * xc))
        r1 = float(np.sum(w * y))
        r2 = float(np.sum(w * xc * y))
        det = s_w * a22 - a12 * a12
        if abs(det) < 1e-14:
            out[i] = r1 / s_w
        else:
            out[i] = (a22 * r1 - a12 * r2) / det
    return out


def stl_decompose(
    y: FloatArray,
    period: int,
    n_inner: int = 2,
    n_outer: int = 1,
    s_win: int | None = None,
    t_win: int | None = None,
    l_win: int | None = None,
) -> dict[str, FloatArray]:
    """STL decomposition into trend + seasonal + residual."""
    v = _as_series(y)
    n = v.size
    if period < 2 or 2 * period >= n:
        raise ValueError("bad period")
    s_win = s_win or max(7, period + 1)
    t_win = t_win or int(1.5 * period / (1 - 1.5 / s_win) + 0.5)
    l_win = l_win or (period + (period % 2 == 0))
    x = np.arange(n, dtype=np.float64)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    for _ in range(max(1, n_outer)):
        # outer loop robustness weights via bisquare on residual
        resid = v - trend - seasonal
        mad = float(np.median(np.abs(resid))) + 1e-12
        u = np.abs(resid) / (6.0 * mad)
        rw = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
        for _ in range(max(1, n_inner)):
            deseason = v - seasonal
            trend = _loess(x, deseason, t_win, rw)
            detrended = v - trend
            # cycle-subseries smoothing
            cycle = np.zeros((period, n))
            for m in range(period):
                idx = np.arange(m, n, period)
                ss = detrended[idx]
                ext = np.concatenate([[ss[0]] * period, ss, [ss[-1]] * period])
                sm = _loess(np.arange(ext.size, dtype=np.float64), ext, s_win)
                cycle[m, idx] = sm[period : period + idx.size]
            seas_raw = np.sum(cycle, axis=0) / 1.0
            # low-pass: MA over period + LOESS
            k = np.ones(period) / period
            lp = np.convolve(np.pad(seas_raw, period // 2, mode="edge"), k, "same")[:n]
            lp = _loess(x, lp, l_win)
            seasonal = seas_raw - lp
            seasonal -= seasonal.mean()
    resid = v - trend - seasonal
    out: dict[str, FloatArray] = {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
    }
    return out


def stl_strength(y: FloatArray, period: int) -> dict[str, float]:
    """Seasonal + trend strength statistics (Wang-Smith-Hyndman)."""
    d = stl_decompose(y, period)
    vr = float(np.var(d["resid"]))
    vs = float(np.var(d["seasonal"] + d["resid"]))
    vt = float(np.var(d["trend"] + d["resid"]))
    out: dict[str, float] = {
        "seasonal_strength": max(0.0, 1.0 - vr / vs),
        "trend_strength": max(0.0, 1.0 - vr / vt),
    }
    return out


def synth_stl(
    seed: int = 20261231 + 339,
    n: int = 480,
    period: int = 24,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC trend + harmonic seasonal + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.02 * t + 3e-5 * t**2
    seasonal = 0.5 * np.sin(2 * np.pi * t / period) + 0.2 * np.sin(4 * np.pi * t / period)
    y = trend + seasonal + 0.15 * rng.standard_normal(n)
    return y.astype(np.float64), trend.astype(np.float64), seasonal.astype(np.float64)


def bench_stl(seed: int = 20261231 + 339) -> dict[str, float]:
    y, trend, seasonal = synth_stl(seed=seed)
    d = stl_decompose(y, period=24)
    var_total = float(np.var(y))
    var_resid = float(np.var(d["resid"]))
    trend_corr = float(np.corrcoef(d["trend"], trend)[0, 1])
    seas_corr = float(np.corrcoef(d["seasonal"], seasonal)[0, 1])
    strength = stl_strength(y, period=24)
    ok = (
        var_resid < 0.25 * var_total
        and trend_corr > 0.98
        and seas_corr > 0.9
        and strength["seasonal_strength"] > 0.8
    )
    out: dict[str, float] = {
        "synthetic_stl_resid_var_ratio": var_resid / var_total,
        "synthetic_stl_trend_corr": trend_corr,
        "synthetic_stl_seasonal_corr": seas_corr,
        "synthetic_stl_seasonal_strength": strength["seasonal_strength"],
        "score": 1.0 if ok else 0.0,
    }
    return out
