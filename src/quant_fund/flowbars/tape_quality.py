"""Trade-tape quality diagnostics.

Scores how clean a tape is before any microstructure estimator touches it:

- ``zero_change_runs`` — run-length distribution of consecutive identical
  prices (staleness proxy: long runs = stale quotes);
- ``discretization_grid`` — the implied price tick (gcd over observed price
  differences, in absolute terms);
- ``outlier_trade_rate`` — fraction of trades with |log return| beyond a
  rolling median ± k·MAD band;
- ``gap_time_stats`` — distribution of inter-trade gaps (index units) with
  median, p95, and max run of empty periods;
- ``tape_quality_report`` — the bundle plus a simple 0-1 cleanliness score
  meant for gating, not marketing.

Honesty: these are descriptive tape checks. They do not score data vendors
or make market claims.

References:
- Hansen, P. R., Lunde, A. (2006). Realized variance and market
  microstructure noise — tape hygiene before estimation.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — data quality gates.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _log_returns(prices: FloatArray) -> FloatArray:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or len(p) < 3:
        raise ValueError("prices must be one-dimensional with >= 3 points")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    return np.diff(np.log(p))


def zero_change_runs(prices: FloatArray) -> dict[str, float]:
    """Run stats over consecutive zero price changes (max/mean run length)."""
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("prices must be one-dimensional with >= 2 points")
    same = (np.diff(p) == 0).astype(np.int64)
    runs = []
    cur = 0
    for v in same:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return {"max_run": 0.0, "mean_run": 0.0, "share_zero": 0.0}
    arr = np.asarray(runs, dtype=np.float64)
    return {
        "max_run": float(np.max(arr)),
        "mean_run": float(np.mean(arr)),
        "share_zero": float(np.mean(same)),
    }


def discretization_grid(prices: FloatArray) -> float:
    """Implied price tick: gcd of nonzero absolute price differences.

    Returns 0.0 when no differences exist.
    """
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("prices must be one-dimensional with >= 2 points")
    diffs = np.abs(np.diff(p))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    # quantize to integers at 1e-6 to get integer gcd
    ints = np.round(diffs / 1e-6).astype(np.int64)
    g = int(math.gcd(*[int(x) for x in ints]))
    return float(g * 1e-6)


def outlier_trade_rate(prices: FloatArray, window: int = 100, k: float = 4.0) -> float:
    """Fraction of log returns outside a rolling median ± k·MAD band."""
    r = _log_returns(prices)
    n = len(r)
    if window < 10:
        raise ValueError("window must be >= 10")
    out = 0
    count = 0
    for t in range(window, n):
        w = r[t - window : t]
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        band = 1.4826 * mad * k
        count += 1
        if abs(r[t] - med) > band:
            out += 1
    return float(out / max(count, 1))


def gap_time_stats(gaps: FloatArray) -> dict[str, float]:
    """Distribution of inter-trade gaps (index or time units)."""
    g = np.asarray(gaps, dtype=np.float64)
    if g.ndim != 1 or len(g) == 0:
        raise ValueError("gaps must be a non-empty one-dimensional array")
    if np.any(g < 0):
        raise ValueError("gaps must be non-negative")
    return {
        "median": float(np.median(g)),
        "p95": float(np.quantile(g, 0.95)),
        "max": float(np.max(g)),
        "mean": float(np.mean(g)),
    }


def tape_quality_report(prices: FloatArray, gaps: FloatArray | None = None) -> dict[str, float]:
    """Bundle the diagnostics plus a simple 0-1 cleanliness score.

    The score starts at 1 and subtracts the stale-run share, outlier rate
    (weighted ×4), and a penalty when the tape has no price grid structure
    (continuous prices) — meant for gating, not for marketing.
    """
    z = zero_change_runs(prices)
    out = outlier_trade_rate(prices)
    grid = discretization_grid(prices)
    score = max(0.0, 1.0 - z["share_zero"] - 4.0 * out)
    if grid == 0.0:
        score = min(score, 0.5)
    out_dict = {
        "max_stale_run": z["max_run"],
        "share_zero_changes": z["share_zero"],
        "outlier_rate": out,
        "discretization_grid": grid,
        "cleanliness": float(score),
    }
    if gaps is not None:
        gs = gap_time_stats(gaps)
        out_dict.update({"gap_median": gs["median"], "gap_p95": gs["p95"]})
    return out_dict
