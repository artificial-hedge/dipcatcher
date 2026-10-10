"""Intraday volatility seasonality (time-of-day factors).

Estimates the deterministic intraday component of volatility — the classic
U-shape — and produces deseasonalised returns for downstream estimators
that assume stationarity within the day.

- ``seasonality_curve`` — minute-of-day volatility factors estimated by
  kernel-weighted averages of squared returns (Andersen-Bollerslev style);
- ``deseasonalize_returns`` — divide returns by the square root of the
  fitted factor, preserving total variance by construction;
- ``seasonal_strength`` — fraction of intraday variance explained by the
  fitted curve (R² of |r|² on the factor).

Honesty: the curve is a descriptive seasonal decomposition of the supplied
series; the U-shape test on synthetic data is a correctness fixture.

References:
- Andersen, T. G., Bollerslev, T. (1997). Intraday periodicity and
  volatility persistence in financial markets — the seasonal factor.
- Boudt, K., Cornelissen, J., Payseur, S. (2017). Intraday seasonality in
  volatility: evidence from international equity markets.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def seasonality_curve(
    returns: FloatArray,
    minute_of_day: IntArray,
    n_minutes: int = 390,
    bandwidth: int = 10,
) -> FloatArray:
    """Minute-of-day volatility factors (mean 1) from squared returns.

    ``minute_of_day`` is an integer index in [0, n_minutes) per return.
    Each minute's factor is a Gaussian-kernel weighted mean of |r|² over
    observations near that minute; the curve is normalised to mean 1.
    """
    r = np.asarray(returns, dtype=np.float64)
    m = np.asarray(minute_of_day, dtype=np.int64)
    if r.ndim != 1 or m.shape != r.shape:
        raise ValueError("returns and minute_of_day must be one-dimensional and equal")
    if np.any((m < 0) | (m >= n_minutes)):
        raise ValueError("minute_of_day out of range")
    if bandwidth < 1:
        raise ValueError("bandwidth must be >= 1")
    r2 = r * r
    grid = np.arange(n_minutes, dtype=np.float64)
    out = np.empty(n_minutes, dtype=np.float64)
    for i in range(n_minutes):
        d2 = (grid[i] - m.astype(np.float64)) ** 2
        w = np.exp(-0.5 * d2 / (bandwidth * bandwidth))
        s = float(w.sum())
        out[i] = float(w @ r2) / s if s > 0 else np.nan
    mean = float(np.nanmean(out))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("seasonality curve is degenerate")
    return np.asarray(out / mean, dtype=np.float64)


def deseasonalize_returns(
    returns: FloatArray,
    minute_of_day: IntArray,
    curve: FloatArray,
) -> FloatArray:
    """Divide returns by √(factor) — variance-preserving deseasonalisation."""
    r = np.asarray(returns, dtype=np.float64)
    m = np.asarray(minute_of_day, dtype=np.int64)
    c = np.asarray(curve, dtype=np.float64)
    if r.shape != m.shape:
        raise ValueError("returns and minute_of_day must have equal shapes")
    if np.any((m < 0) | (m >= len(c))):
        raise ValueError("minute_of_day out of range for the curve")
    if np.any(c <= 0):
        raise ValueError("curve factors must be positive")
    return np.asarray(r / np.sqrt(c[m]), dtype=np.float64)


def seasonal_strength(returns: FloatArray, minute_of_day: IntArray, curve: FloatArray) -> float:
    """R² of squared returns on the fitted seasonal factor, in [0, 1].

    Regresses r² on the factor through the origin (β = Σ c·r² / Σ c²) and
    reports the fraction of r² variance explained.
    """
    r = np.asarray(returns, dtype=np.float64)
    m = np.asarray(minute_of_day, dtype=np.int64)
    c = np.asarray(curve, dtype=np.float64)
    if r.shape != m.shape:
        raise ValueError("returns and minute_of_day must have equal shapes")
    if np.any((m < 0) | (m >= len(c))):
        raise ValueError("minute_of_day out of range for the curve")
    r2 = r * r
    factor = c[m]
    ss_tot = float(np.sum((r2 - r2.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    beta = float(np.dot(factor, r2) / max(np.dot(factor, factor), 1e-12))
    ss_res = float(np.sum((r2 - beta * factor) ** 2))
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
