"""Quantile-spread (bucket) diagnostics for a cross-sectional signal.

Sorts assets into K buckets by signal each period and studies the forward
return spread across buckets — the classic "Q1 vs QK" monotonicity picture:

- ``bucket_returns`` — mean forward return per bucket per period;
- ``spread_curve`` — long-short spread QK − Q1 per period and its mean;
- ``monotonicity_score`` — fraction of adjacent bucket-mean pairs that
  increase in the signal direction;
- ``bucket_trend_ic`` — Spearman IC between the bucket index (1..K) and the
  bucket mean return (Jonckheere-style trend statistic).

Honesty: bucket spreads are descriptive diagnostics on the supplied panel,
never performance headlines.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — ranking and
  bucket analysis of signals.
- Jonckheere, A. R. (1954). A distribution-free k-sample test against
  ordered alternatives — the trend test spirit.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bucket_returns(pred: FloatArray, fwd: FloatArray, k: int = 5) -> FloatArray:
    """Mean forward return per bucket per period.

    ``pred`` is (T, N); ``fwd`` is (T, N). Assets are ranked by signal each
    period and split into k equal-frequency buckets. Returns (T, k) with
    NaN where a bucket is empty.
    """
    pred = np.asarray(pred, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    if pred.shape != fwd.shape or pred.ndim != 2:
        raise ValueError("pred and fwd must be (T, N) of equal shape")
    if k < 2:
        raise ValueError("k must be >= 2")
    t_total, n = pred.shape
    if n < k:
        raise ValueError("need at least k assets")
    out = np.full((t_total, k), np.nan, dtype=np.float64)
    for t in range(t_total):
        x = pred[t]
        y = fwd[t]
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < k:
            continue
        ranks = np.argsort(np.argsort(x[valid]))
        edges = np.linspace(0, np.sum(valid), k + 1).astype(np.int64)
        for b in range(k):
            sel = (ranks >= edges[b]) & (ranks < edges[b + 1])
            if np.any(sel):
                out[t, b] = float(np.mean(y[valid][sel]))
    return out


def spread_curve(buckets: FloatArray) -> FloatArray:
    """Long-short spread last-minus-first bucket per period."""
    b = np.asarray(buckets, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] < 2:
        raise ValueError("buckets must be (T, k) with k >= 2")
    return np.asarray(b[:, -1] - b[:, 0], dtype=np.float64)


def monotonicity_score(buckets: FloatArray) -> float:
    """Fraction of adjacent bucket-mean pairs increasing (NaN-aware)."""
    b = np.asarray(buckets, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] < 2:
        raise ValueError("buckets must be (T, k) with k >= 2")
    means = np.nanmean(b, axis=0)
    if np.any(np.isnan(means)):
        raise ValueError("buckets must be non-empty in every bucket")
    diffs = np.diff(means)
    return float(np.mean(diffs > 0))


def bucket_trend_ic(buckets: FloatArray) -> dict[str, float]:
    """Spearman IC between bucket index 1..k and mean bucket return."""
    b = np.asarray(buckets, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] < 3:
        raise ValueError("buckets must be (T, k) with k >= 3")
    means = np.nanmean(b, axis=0)
    if np.any(np.isnan(means)):
        raise ValueError("buckets must be non-empty in every bucket")
    idx = np.arange(1, b.shape[1] + 1, dtype=np.float64)
    r_idx = np.argsort(np.argsort(idx)).astype(np.float64)
    r_ret = np.argsort(np.argsort(means)).astype(np.float64)
    ic = float(np.corrcoef(r_idx, r_ret)[0, 1])
    return {"trend_ic": ic, "k": float(b.shape[1])}
