"""Merging member prediction intervals into coherent ensemble bands.

Given per-member intervals, produce ensemble bands with controlled marginal
coverage and score them with the Winkler interval score:

- ``bonferroni_union`` — union band [min lower, max upper] across members;
  marginal coverage is at least 1 − Σ α_k by Bonferroni when member coverages
  hold jointly;
- ``quantile_band_average`` — per-side quantile averaging of member bands;
- ``winkler_score`` — the interval (Winkler) score rewarding narrow bands
  that cover, penalising misses distance-linearly;
- ``band_coverage`` — empirical hit rate of a band against realised values.

Honesty: coverage statements are marginal and assume member validity; the
Bonferroni bound is a statement about simultaneous coverage.

References:
- Winkler, R. L. (1972). A decision-theoretic approach to interval
  estimation — the interval score.
- Dunn, O. J. (1961). Multiple comparisons among means — the Bonferroni
  union argument.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_band(lower: FloatArray, upper: FloatArray) -> tuple[FloatArray, FloatArray]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("lower and upper must have equal shapes")
    if np.any(hi < lo):
        raise ValueError("upper must be >= lower everywhere")
    return lo, hi


def bonferroni_union(
    member_lower: FloatArray,
    member_upper: FloatArray,
) -> dict[str, FloatArray]:
    """Union band across members: [min_k lower_k, max_k upper_k].

    ``member_lower``/``member_upper`` are (M, T) with member axis first.
    """
    lo = np.asarray(member_lower, dtype=np.float64)
    hi = np.asarray(member_upper, dtype=np.float64)
    if lo.ndim != 2 or hi.shape != lo.shape:
        raise ValueError("member bands must be (M, T) of equal shape")
    if np.any(hi < lo):
        raise ValueError("each member band needs upper >= lower")
    return {
        "lower": np.asarray(np.min(lo, axis=0), dtype=np.float64),
        "upper": np.asarray(np.max(hi, axis=0), dtype=np.float64),
    }


def quantile_band_average(
    member_lower: FloatArray,
    member_upper: FloatArray,
    weights: FloatArray | None = None,
) -> dict[str, FloatArray]:
    """Weighted quantile-average band per side (no Bonferroni widening)."""
    lo = np.asarray(member_lower, dtype=np.float64)
    hi = np.asarray(member_upper, dtype=np.float64)
    if lo.ndim != 2 or hi.shape != lo.shape:
        raise ValueError("member bands must be (M, T) of equal shape")
    m = lo.shape[0]
    if weights is None:
        w: FloatArray = np.full(m, 1.0 / m)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (m,) or np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("weights must be (M,) non-negative with positive sum")
        w = w / float(w.sum())
    out_lo = cast(FloatArray, np.tensordot(w, lo, axes=([0], [0])))
    out_hi = cast(FloatArray, np.tensordot(w, hi, axes=([0], [0])))
    return {
        "lower": np.asarray(np.minimum(out_lo, out_hi), dtype=np.float64),
        "upper": np.asarray(np.maximum(out_lo, out_hi), dtype=np.float64),
    }


def band_coverage(y: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """Empirical fraction of observations inside [lower, upper] (NaN-free)."""
    lo, hi = _check_band(lower, upper)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != lo.shape:
        raise ValueError("y must match the band shapes")
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        raise ValueError("no finite observations to score")
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def winkler_score(
    y: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    alpha: float,
) -> FloatArray:
    """Per-observation Winkler interval score.

    score = (upper − lower) + (2/α)·(lower − y)·1{y < lower}
          + (2/α)·(y − upper)·1{y > upper}. Lower is better; the penalty is
    distance-linear in the miss size.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    lo, hi = _check_band(lower, upper)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != lo.shape:
        raise ValueError("y must match the band shapes")
    width = hi - lo
    miss_low = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    miss_high = np.where(y > hi, (2.0 / alpha) * (y - hi), 0.0)
    return np.asarray(width + miss_low + miss_high, dtype=np.float64)
