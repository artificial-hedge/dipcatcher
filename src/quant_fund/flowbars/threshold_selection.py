"""Data-driven selection of the dollar-bar threshold.

The dollar-bar threshold trades estimator variance against sampling
frequency: too many tiny bars → RV is noisy; too few bars → coarse. This
module scores candidate thresholds by the *variance of the per-bar return
variance* — over-dispersion of bar volatilities signals threshold
mismatch — and picks the threshold minimising it.

- ``threshold_profile`` — per-threshold diagnostics: bar count, mean bar
  variance, dispersion (std of per-bar |r|), and the selection score;
- ``select_threshold`` — argmin of the dispersion score with monotone
  fallback when the profile is flat.

Honesty: the criterion is a heuristic stabiliser (variance-of-variance),
not an optimality proof; synthetic fixtures validate the mechanics.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — bar threshold choice as a bias/variance dial.
- Andersen, T. G., Bollerslev, T. (1998). Answering the skeptics — RV
  behaviour under sampling frequency.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def threshold_profile(
    dollar: FloatArray,
    prices: FloatArray,
    thresholds: FloatArray,
) -> dict[str, FloatArray]:
    """Score each candidate dollar threshold on bar-return dispersion.

    For every threshold: build bars, compute per-bar |log returns|, and
    report count, mean, and coefficient of variation. The selection score
    is the CV of per-bar absolute returns (lower = more homogeneous bars).
    """
    d = np.asarray(dollar, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    if d.shape != p.shape or d.ndim != 1:
        raise ValueError("dollar and prices must be one-dimensional arrays of equal shape")
    ts = np.asarray(thresholds, dtype=np.float64)
    if len(ts) == 0 or np.any(ts <= 0):
        raise ValueError("thresholds must be positive")
    counts = np.empty(len(ts), dtype=np.float64)
    means = np.empty(len(ts), dtype=np.float64)
    cvs = np.empty(len(ts), dtype=np.float64)
    log_p = np.log(p)
    cum_d = np.cumsum(d)
    for i, t in enumerate(ts):
        ids = np.floor(cum_d / t).astype(np.int64)
        _, dense = np.unique(ids, return_inverse=True)
        ends = np.append(np.flatnonzero(np.diff(dense)), len(dense) - 1)
        starts = np.concatenate(([0], ends[:-1] + 1))
        seg_var = np.empty(len(ends), dtype=np.float64)
        for b in range(len(ends)):
            lp = log_p[starts[b] : ends[b] + 1]
            r = np.diff(lp)
            seg_var[b] = float(np.sum(r * r)) / max(len(r), 1)
        absr = np.sqrt(seg_var)
        counts[i] = float(len(ends))
        means[i] = float(np.mean(absr))
        cvs[i] = float(np.std(absr, ddof=1) / max(np.mean(absr), 1e-12))
    return {"threshold": ts, "n_bars": counts, "mean_abs_return": means, "cv": cvs}


def select_threshold(
    dollar: FloatArray,
    prices: FloatArray,
    thresholds: FloatArray,
) -> dict[str, float]:
    """Threshold with the lowest bar-dispersion score (first-min tie-break)."""
    prof = threshold_profile(dollar, prices, thresholds)
    cv = prof["cv"]
    i = int(np.nanargmin(cv))
    return {
        "threshold": float(prof["threshold"][i]),
        "cv": float(cv[i]),
        "n_bars": float(prof["n_bars"][i]),
    }
