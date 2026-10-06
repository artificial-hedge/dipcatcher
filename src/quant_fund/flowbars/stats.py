"""Bar-return diagnostics: serial dependence and shape statistics.

Computes the statistics used to compare bar constructions on equal footing:
lag-1 serial correlation, the variance ratio at horizon 2, Durbin-Watson,
skewness and excess kurtosis, plus a head-to-head ``compare_bar_returns``.
Under i.i.d. returns the variance ratio is 1 and the serial correlation is 0;
departures indicate dependence left in the bar series (which is what
information-driven bars are designed to reduce, relative to time bars).

Honesty: these are descriptive statistics of the supplied series. They say
nothing about future returns or tradability.

References:
- Lo, A. W., MacKinlay, A. C. (1988). Stock market prices do not follow
  random walks: evidence from a simple specification test — variance ratio.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — comparing bar constructions via return statistics.

Composition: pure numpy; deterministic; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _clean(r: FloatArray) -> FloatArray:
    out = np.asarray(r, dtype=np.float64)
    out = out[np.isfinite(out)]
    if len(out) < 3:
        raise ValueError("need at least 3 finite returns")
    return out


def lag1_serial_corr(r: FloatArray) -> float:
    """Pearson correlation between consecutive returns."""
    x = _clean(r)
    a = x[:-1] - x[:-1].mean()
    b = x[1:] - x[1:].mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def variance_ratio(r: FloatArray, horizon: int = 2) -> float:
    """Lo-MacKinlay variance ratio Var(Σ_h r) / (h · Var(r))."""
    x = _clean(r)
    if horizon < 2:
        raise ValueError("horizon must be >= 2")
    if len(x) <= horizon:
        raise ValueError("series too short for the requested horizon")
    var_1 = float(np.var(x, ddof=1))
    if var_1 <= 0:
        return float("nan")
    agg = np.sum(np.lib.stride_tricks.sliding_window_view(x, horizon), axis=1)
    var_h = float(np.var(agg, ddof=1))
    return float(var_h / (horizon * var_1))


def durbin_watson(r: FloatArray) -> float:
    """Durbin-Watson statistic Σ(e_t − e_{t−1})² / Σ e_t²."""
    x = _clean(r)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    d = np.diff(x)
    return float(np.sum(d * d) / denom)


def bar_return_stats(r: FloatArray) -> dict[str, float]:
    """Full diagnostic dictionary for a bar-return series."""
    x = _clean(r)
    n = float(len(x))
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        skew = float("nan")
        exkurt = float("nan")
    else:
        z = (x - mu) / sd
        skew = float(np.mean(z**3))
        exkurt = float(np.mean(z**4) - 3.0)
    return {
        "n": n,
        "mean": mu,
        "std": sd,
        "skew": skew,
        "excess_kurtosis": exkurt,
        "serial_corr_lag1": lag1_serial_corr(x),
        "variance_ratio_2": variance_ratio(x, horizon=2),
        "durbin_watson": durbin_watson(x),
    }


def bars_per_period(ids: IntArray, period_length: int) -> IntArray:
    """Number of bars *closing* in each fixed-length period (e.g. per day)."""
    ids = np.asarray(ids, dtype=np.int64)
    if period_length <= 0:
        raise ValueError("period_length must be positive")
    if len(ids) == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(ids))
    closes = np.append(change, len(ids) - 1)
    n_periods = int(np.ceil(len(ids) / period_length))
    out = np.zeros(n_periods, dtype=np.int64)
    np.add.at(out, closes // period_length, 1)
    return out


def compare_bar_returns(r_time: FloatArray, r_alt: FloatArray) -> dict[str, dict[str, float]]:
    """Side-by-side stats for two bar constructions (e.g. time vs dollar).

    ``reduction_*`` entries report (stat_time − stat_alt) / |stat_time| for
    serial correlation and excess kurtosis — positive means the alternative
    bar reduced the dependence/shape distortion.
    """
    s_time = bar_return_stats(r_time)
    s_alt = bar_return_stats(r_alt)

    def _reduction(key: str) -> float:
        a = s_time[key]
        b = s_alt[key]
        if not np.isfinite(a) or abs(a) < 1e-12:
            return float("nan")
        return float((a - b) / abs(a))

    return {
        "time": s_time,
        "alternative": s_alt,
        "reduction_serial_corr": _reduction("serial_corr_lag1"),
        "reduction_excess_kurtosis": _reduction("excess_kurtosis"),
    }
