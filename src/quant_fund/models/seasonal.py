"""Classical seasonal decomposition and seasonal strength measures (SYNTHETIC).

- ``seasonal_decompose``: classical (centered-MA) additive or
  multiplicative decomposition into trend / seasonal / residual.
- ``seasonal_strength`` / ``trend_strength``: Wang, Smith & Hyndman
  (2006) strength measures in [0, 1].

Fail-closed: series shorter than 2 full periods, non-finite data,
non-positive values under the multiplicative model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(y: Array, period: int) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if period < 2 or yy.size < 2 * period:
        raise ValueError("need >= 2 full periods")
    if not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    return yy


def _centered_ma(y: Array, period: int) -> Array:
    """Centered moving average aligned to y (NaN at boundaries)."""
    n = y.size
    trend = np.full(n, np.nan)
    if period % 2 == 0:
        # 2 x period MA
        ma = np.convolve(y, np.ones(period) / period, mode="valid")
        trend_vals = (ma[:-1] + ma[1:]) / 2.0
        start = period // 2
        trend[start : start + trend_vals.size] = trend_vals
    else:
        ma = np.convolve(y, np.ones(period) / period, mode="valid")
        start = period // 2
        trend[start : start + ma.size] = ma
    return trend


def seasonal_decompose(y: Array, period: int, model: str = "additive") -> dict[str, Array]:
    """Classical decomposition. Returns trend, seasonal, resid (NaN where
    the trend is undefined at boundaries)."""
    yy = _check(y, period)
    if model not in ("additive", "multiplicative"):
        raise ValueError("model must be additive|multiplicative")
    if model == "multiplicative" and (yy <= 0).any():
        raise ValueError("multiplicative model requires y > 0")
    n = yy.size
    trend = _centered_ma(yy, period)
    valid = np.isfinite(trend)
    detrended = np.full(n, np.nan)
    detrended[valid] = (
        (yy[valid] - trend[valid]) if model == "additive" else (yy[valid] / trend[valid])
    )
    # average detrended values within each seasonal position
    indices = np.full(period, np.nan)
    for k in range(period):
        vals = detrended[k::period]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            indices[k] = vals.mean()
    if model == "additive":
        indices -= indices.mean()  # enforce sum-to-zero
        seasonal = np.tile(indices, int(np.ceil(n / period)))[:n]
        resid = np.full(n, np.nan)
        resid[valid] = yy[valid] - trend[valid] - seasonal[valid]
    else:
        indices /= np.exp(np.log(indices).mean())  # geometric normalization
        seasonal = np.tile(indices, int(np.ceil(n / period)))[:n]
        resid = np.full(n, np.nan)
        resid[valid] = yy[valid] / (trend[valid] * seasonal[valid])
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "indices": indices}


def seasonal_strength(y: Array, period: int, model: str = "additive") -> float:
    """WSH seasonal strength: max(0, 1 - Var(resid)/Var(detrended))."""
    out = seasonal_decompose(y, period, model)
    resid = np.asarray(out["resid"])
    trend = np.asarray(out["trend"])
    yy = _check(y, period)
    valid = np.isfinite(resid)
    if valid.sum() < period:
        raise ValueError("insufficient interior observations")
    detrended = yy[valid] - trend[valid] if model == "additive" else yy[valid] / trend[valid]
    v_resid = float(np.var(resid[valid]))
    v_detr = float(np.var(detrended))
    return float(max(0.0, 1.0 - v_resid / max(v_detr, 1e-14)))


def trend_strength(y: Array, period: int, model: str = "additive") -> float:
    """WSH trend strength: max(0, 1 - Var(resid)/Var(deseasonalized))."""
    out = seasonal_decompose(y, period, model)
    resid = np.asarray(out["resid"])
    seasonal = np.asarray(out["seasonal"])
    yy = _check(y, period)
    valid = np.isfinite(resid)
    if valid.sum() < period:
        raise ValueError("insufficient interior observations")
    deseas = yy[valid] - seasonal[valid] if model == "additive" else yy[valid] / seasonal[valid]
    v_resid = float(np.var(resid[valid]))
    v_ds = float(np.var(deseas))
    return float(max(0.0, 1.0 - v_resid / max(v_ds, 1e-14)))


def bench_seasonal(seed: int = 20261231 + 810) -> dict[str, float]:
    """Classical decomposition oracle: planted sinusoid + linear trend +
    noise must decompose cleanly; strength measures must rank the
    seasonal series far above a noise-only series."""
    rng = np.random.default_rng(seed)
    n, period = 240, 12
    t = np.arange(n)
    seasonal_true = 3.0 * np.sin(2.0 * np.pi * t / period)
    trend_true = 0.02 * t
    y = trend_true + seasonal_true + 0.3 * rng.standard_normal(n)
    y_noise = 0.3 * rng.standard_normal(n)
    dec = seasonal_decompose(y, period)
    s_hat = np.asarray(dec["seasonal"])
    corr_s = float(np.corrcoef(s_hat[period:-period], seasonal_true[period:-period])[0, 1])
    resid = np.asarray(dec["resid"])
    recon = np.asarray(dec["trend"]) + s_hat + resid
    ss = seasonal_strength(y, period)
    ss_noise = seasonal_strength(y_noise, period)
    ts = trend_strength(y, period)
    checks = [
        corr_s > 0.9,
        float(np.nanmax(np.abs(recon - y))) < 1e-8,
        ss > 0.8,
        ss_noise < ss - 0.5,
        0.0 <= ts <= 1.0,
    ]
    if not all(checks):
        raise ValueError("seasonal decomposition oracle failed")
    return {
        "synthetic_seasonal_corr": corr_s,
        "synthetic_seasonal_strength": ss,
        "synthetic_seasonal_noise_strength": ss_noise,
        "synthetic_seasonal_score": float(sum(bool(c) for c in checks) / len(checks)),
    }
