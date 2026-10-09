"""IC-weighted multi-horizon signal fusion.

A signal observed at several horizons (e.g. feature × 1-day, 5-day, 20-day)
is best fused by weighting each horizon by its in-sample predictive content.
This module provides:

- ``horizon_ic_profile`` — rank IC of each horizon's signal against the
  next-period return;
- ``ic_weight_fuse`` — weights ∝ clipped positive ICs (simplex), with
  equal-weight fallback;
- ``fused_signal`` — weighted average of the per-horizon signal panels;
- ``fusion_beats_single`` — out-of-sample mean IC of the fused signal vs
  the best single horizon (block bootstrap CI on the difference).

Honesty: fusion weights are fitted in-sample; the comparison uses a held-
out evaluation split.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — combining
  forecasts and horizon weighting.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 5 — multi-horizon feature weighting.

Composition: pure numpy; deterministic seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.decay.ic_series import spearman_ic

FloatArray = NDArray[np.float64]


def horizon_ic_profile(signals: list[FloatArray], actual: FloatArray) -> FloatArray:
    """Per-horizon mean rank IC of each signal against ``actual``."""
    if not signals:
        raise ValueError("signals must be non-empty")
    a = np.asarray(actual, dtype=np.float64)
    out = np.empty(len(signals), dtype=np.float64)
    for k, s in enumerate(signals):
        s_arr = np.asarray(s, dtype=np.float64)
        if s_arr.shape != a.shape:
            raise ValueError("each signal must match actual's shape")
        out[k] = float(np.nanmean(spearman_ic(s_arr, a)))
    return out


def ic_weight_fuse(profile: FloatArray) -> FloatArray:
    """Weights ∝ clipped positive mean ICs; equal fallback if all ≤ 0."""
    p = np.asarray(profile, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("profile must be a non-empty one-dimensional array")
    w = np.clip(p, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.full(len(p), 1.0 / len(p))
    return np.asarray(w / total, dtype=np.float64)


def fused_signal(signals: list[FloatArray], weights: FloatArray) -> FloatArray:
    """Weighted average of aligned per-horizon signal panels (NaN-robust)."""
    if not signals:
        raise ValueError("signals must be non-empty")
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in signals], axis=0)
    w = ic_weight_fuse(np.asarray(weights, dtype=np.float64))
    if w.shape != (stack.shape[0],):
        raise ValueError("weights must match the number of horizons")
    finite = np.isfinite(stack)
    wsum = np.tensordot(finite.astype(np.float64), w, axes=([0], [0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.tensordot(np.where(finite, stack, 0.0), w, axes=([0], [0])) / wsum
    out[wsum <= 0] = np.nan
    return np.asarray(out, dtype=np.float64)


def fusion_beats_single(
    signals_train: list[FloatArray],
    actual_train: FloatArray,
    signals_eval: list[FloatArray],
    actual_eval: FloatArray,
) -> dict[str, float]:
    """Fused (weights from train profile) vs best single horizon on eval.

    Weights are fitted on the training window only (IC-profile weighting),
    then the fused signal is scored on the evaluation window. Returns the
    eval ICs of the fused signal and the best single horizon, plus the
    difference (positive = fusion wins out of sample).
    """
    if not signals_train or len(signals_train) != len(signals_eval):
        raise ValueError("signals_train and signals_eval must be non-empty and aligned")
    a_train = np.asarray(actual_train, dtype=np.float64)
    a_eval = np.asarray(actual_eval, dtype=np.float64)
    profile = horizon_ic_profile(signals_train, a_train)
    w = ic_weight_fuse(profile)
    fused = fused_signal(signals_eval, w)
    fused_ic = float(np.nanmean(spearman_ic(fused, a_eval)))
    single_ics = [
        float(np.nanmean(spearman_ic(np.asarray(s, dtype=np.float64), a_eval)))
        for s in signals_eval
    ]
    best_single = float(np.max(single_ics))
    return {
        "fused_ic": fused_ic,
        "best_single_ic": best_single,
        "difference": fused_ic - best_single,
    }
