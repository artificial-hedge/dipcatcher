"""Decay-aware combination of multiple signals.

Turns per-signal IC profiles into portfolio-style signal weights:

- winsorized IC weights: mean-IC winsorized across signals, negatives
  clipped, normalised;
- decay-profile weights: w_i ∝ Σ_k |IC_i(k)| · exp(−k / hl_i), so signals
  with slower decay and stronger horizon curves get more mass;
- combine_signals: NaN-robust weighted average with renormalisation over
  available signals per (t, n) cell.

Honesty: weights are fitted on the supplied panel; no out-of-sample claim.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — combining
  forecasts and the fundamental law.
- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts —
  convex combination weights.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def winsorized_ic_weights(
    ic_means: FloatArray,
    *,
    limits: tuple[float, float] = (0.05, 0.95),
) -> FloatArray:
    """Weights ∝ winsorized mean IC (across signals), negatives clipped."""
    x = np.asarray(ic_means, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("ic_means must be a non-empty one-dimensional array")
    lo, hi = limits
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("limits must satisfy 0 <= lo < hi <= 1")
    qlo, qhi = np.quantile(x, [lo, hi])
    w = np.clip(x, qlo, qhi)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.full(len(x), 1.0 / len(x), dtype=np.float64)
    return np.asarray(w / total, dtype=np.float64)


def clip_negative_weights(w: FloatArray) -> FloatArray:
    """Clip negatives and renormalise to sum 1 (equal fallback)."""
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.full(len(w), 1.0 / len(w), dtype=np.float64)
    return np.asarray(w / total, dtype=np.float64)


def decay_profile_weights(
    ic_curves: FloatArray,
    half_lives: FloatArray,
) -> FloatArray:
    """Weights ∝ mean |IC_i(k)| × Σ_k exp(−k / hl_i), normalised to sum 1.

    ``ic_curves`` is (S, K) per-signal IC at lags k = 1..K;
    ``half_lives`` is (S,). Non-finite or non-positive half-lives fall back
    to the flat decay factor 1 (no extra reward for non-decaying estimates).
    """
    curves = np.asarray(ic_curves, dtype=np.float64)
    hls = np.asarray(half_lives, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError("ic_curves must be (S, K)")
    s, k = curves.shape
    if hls.shape != (s,):
        raise ValueError("half_lives must be (S,)")
    lags = np.arange(1, k + 1, dtype=np.float64)
    decay = np.ones(s, dtype=np.float64)
    for i in range(s):
        hl = float(hls[i])
        if np.isfinite(hl) and hl > 0:
            decay[i] = float(np.sum(np.exp(-lags / hl)))
        else:
            decay[i] = 1.0
    mass = np.nanmean(np.abs(curves), axis=1) * decay
    mass = np.where(np.isfinite(mass), np.maximum(mass, 0.0), 0.0)
    total = float(mass.sum())
    if total <= 0:
        return np.full(s, 1.0 / s, dtype=np.float64)
    return np.asarray(mass / total, dtype=np.float64)


def combine_signals(signals: list[FloatArray], weights: FloatArray) -> FloatArray:
    """Weighted average of aligned signal panels, NaN-robust per cell.

    ``signals`` is a list of (T, N) arrays; ``weights`` is (S,). For each
    cell the average uses only signals with finite values there, reweighted
    to sum 1. All-NaN cells produce NaN.
    """
    if not signals:
        raise ValueError("signals must be non-empty")
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in signals], axis=0)
    shape = stack.shape[1:]
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (stack.shape[0],):
        raise ValueError("weights must be (S,)")
    w = clip_negative_weights(w)
    finite = np.isfinite(stack)
    wsum = np.tensordot(finite.astype(np.float64), w, axes=([0], [0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.tensordot(np.where(finite, stack, 0.0), w, axes=([0], [0])) / wsum
    out[wsum <= 0] = np.nan
    return np.asarray(out.reshape(shape), dtype=np.float64)
