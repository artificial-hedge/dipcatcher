"""Trimmed and winsorized ensembles of quantile forecasts.

Pointwise (per quantile level) robust aggregation across members:

- trimmed mean: sort members at each quantile level, drop the top and
  bottom fractions, average the rest (falls back to the median when the
  trim would remove everything);
- median ensemble: the per-level median member;
- winsorize_members: clamp each member to the cross-member quantile band.

Robust aggregation bounds the damage one mis-calibrated member can do while
keeping most of the ensemble's information — measured here with pinball/CRPS
on the supplied evaluation set.

Honesty: gains are score improvements on supplied data, not live claims.

References:
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules.
- Huber, P. J. (1964). Robust estimation of a location parameter — trimmed
  means as robust location.

Composition: pure numpy; evaluation via quant_fund.metrics.scoring is done
by callers/tests, keeping this module focused on aggregation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def trimmed_mean_ensemble(member_quantiles: FloatArray, trim: float) -> FloatArray:
    """Per-quantile trimmed mean across members.

    ``member_quantiles`` is (M, ..., Q); ``trim`` in [0, 0.5) is the fraction
    dropped from each side.
    """
    q = np.asarray(member_quantiles, dtype=np.float64)
    if q.ndim < 2:
        raise ValueError("member_quantiles must have a member axis and a quantile axis")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    m = q.shape[0]
    k = int(np.floor(trim * m))
    if 2 * k >= m:
        return median_ensemble(q)
    s = np.sort(q, axis=0)
    kept = s[k : m - k]
    return np.asarray(np.mean(kept, axis=0), dtype=np.float64)


def median_ensemble(member_quantiles: FloatArray) -> FloatArray:
    """Per-quantile median across members."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    if q.ndim < 2:
        raise ValueError("member_quantiles must have a member axis and a quantile axis")
    return np.asarray(np.median(q, axis=0), dtype=np.float64)


def winsorize_members(
    member_quantiles: FloatArray,
    limits: tuple[float, float] = (0.05, 0.95),
) -> FloatArray:
    """Clamp each member to the cross-member quantile band per quantile level."""
    q = np.asarray(member_quantiles, dtype=np.float64)
    if q.ndim < 2:
        raise ValueError("member_quantiles must have a member axis and a quantile axis")
    lo, hi = limits
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("limits must satisfy 0 <= lo < hi <= 1")
    qlo, qhi = np.quantile(q, [lo, hi], axis=0)
    return np.clip(q, qlo, qhi)
