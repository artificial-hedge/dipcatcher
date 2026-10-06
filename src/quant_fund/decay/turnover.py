"""Signal turnover and cost-adjusted net IC.

Turnover measures how much a signal-driven weight vector changes per period;
combined with the horizon-IC curve it converts predictive content into a
*cost-adjusted* net curve, net_ic(k) = ic(k) − cost · turnover(k). The
breakeven cost where cumulative net IC vanishes is the gross-to-net capacity
summary used to compare signals on equal footing.

Honesty: cost units are abstract (IC points per unit turnover); no actual
commission schedule or execution-quality claim is implied.

References:
- Grinold, R., Kahn, R. (2000). *Active Portfolio Management* — turnover,
  cost, and the fundamental law with costs.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 13 — position sizing and capacity reasoning.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def signal_weights(pred: FloatArray, gross: float = 1.0) -> FloatArray:
    """Long-short weights ∝ demeaned cross-sectional signal, gross = Σ|w|.

    Rows that are constant (no cross-sectional dispersion) map to zero
    weights. NaN predictions are excluded and the row is renormalised.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError("pred must be (T, N)")
    if gross <= 0:
        raise ValueError("gross must be positive")
    out = np.zeros_like(pred)
    for t in range(pred.shape[0]):
        row = pred[t]
        valid = np.isfinite(row)
        if not np.any(valid):
            continue
        x = np.where(valid, row - np.nanmean(row), 0.0)
        l1 = float(np.sum(np.abs(x)))
        if l1 <= 0:
            continue
        out[t] = x * (gross / l1)
    return out


def signal_turnover(weights: FloatArray) -> FloatArray:
    """Per-period turnover ½Σ|w_t − w_{t−1}|; first period uses w_0 as entry."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("weights must be (T, N)")
    if len(w) == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(len(w), dtype=np.float64)
    out[0] = 0.5 * float(np.sum(np.abs(w[0])))
    if len(w) > 1:
        out[1:] = 0.5 * np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    return out


def cost_adjusted_ic(ic: FloatArray, turnover: FloatArray, cost: float) -> FloatArray:
    """Net IC after a linear cost drag: ic_t − cost · turnover_t."""
    ic = np.asarray(ic, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)
    if ic.shape != turnover.shape:
        raise ValueError("ic and turnover must have equal shapes")
    if cost < 0:
        raise ValueError("cost must be non-negative")
    return np.asarray(ic - cost * turnover, dtype=np.float64)


def breakeven_cost(ic: FloatArray, turnover: FloatArray) -> float:
    """Cost level at which cumulative net IC is zero (Σic / Σturnover)."""
    ic = np.asarray(ic, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)
    if ic.shape != turnover.shape:
        raise ValueError("ic and turnover must have equal shapes")
    ic_valid = ic[np.isfinite(ic)]
    to_valid = turnover[np.isfinite(turnover)]
    denom = float(np.sum(np.abs(to_valid)))
    if denom <= 0:
        return float("inf")
    return float(np.sum(ic_valid) / denom)
