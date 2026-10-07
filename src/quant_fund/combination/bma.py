"""Pseudo-Bayesian model averaging over forecast members.

Weights members by exponentially weighted cumulative proper scores
(log-score or CRPS): W_{t,k} ∝ exp((Σ_s γ^{t−s} s_{s,k}) / temperature).
With γ = 1 this is the Bates–Granger style exponential weighting; γ < 1
forgets old scores so weights can track a drifting best member.

Honesty: weights reflect past scores on the supplied evaluation stream;
concentration on one member is descriptive, not proof of future dominance.

References:
- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules —
  only proper scores may drive the pseudo-likelihood.
- McLean, D. J., Pontiff, J. (2016). Does academic research destroy stock
  return predictability? — forgetting-factor weighting motivation.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def log_score_weights(
    scores: FloatArray,
    temperature: float = 1.0,
    forgetting: float = 1.0,
) -> FloatArray:
    """Final pseudo-BMA weight path from a (T, K) score stream (higher better)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("scores must be (T, K)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 < forgetting <= 1.0:
        raise ValueError("forgetting must be in (0, 1]")
    t_total, k = s.shape
    weights = np.empty((t_total, k), dtype=np.float64)
    for t in range(t_total):
        ages = t - np.arange(t + 1)
        w_age = forgetting**ages
        z = (w_age @ s[: t + 1]) / temperature
        z = z - float(np.max(z))
        p = np.exp(z)
        weights[t] = p / float(p.sum())
    return weights


def bma_combine(member_preds: FloatArray, weights: FloatArray) -> FloatArray:
    """Combine member predictions (T, K) with weights (T, K)."""
    m = np.asarray(member_preds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if m.ndim != 2 or w.shape != m.shape:
        raise ValueError("member_preds and weights must both be (T, K)")
    combined = np.sum(m * w, axis=1)
    return np.asarray(combined, dtype=np.float64)
