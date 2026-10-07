"""Variance-forecast combination under QLIKE and squared error.

Two classical combination rules:

- ``min_var_weights``: Markowitz combination of point forecasts using their
  error covariance, w = Σ⁻¹1 / (1ᵀΣ⁻¹1) — the forecast combination puzzle
  classic; with ridge regularisation and shrinkage toward equal weights;
- ``crps_weights``: pseudo-likelihood softmax weights from member CRPS or
  QLIKE scores (lower score → higher weight, temperature-scaled).

Honesty: scores are QLIKE/CRPS on the supplied evaluation window; no
volatility-timing or performance claim.

References:
- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts.
- Stock, J. H., Watson, M. W. (2004). Combination forecasts of output
  growth — the combination puzzle and shrinkage.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies — QLIKE as a robust loss for variance forecasts.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def min_var_weights(errors: FloatArray, ridge: float = 1e-6) -> FloatArray:
    """w = Σ⁻¹1 / (1ᵀΣ⁻¹1) from the member error covariance (unconstrained).

    Negative weights are allowed (the classic Granger–Ramanathan variant);
    use ``shrink_to_equal`` when the estimate is too noisy.
    """
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError("errors must be (T, K)")
    t, k = e.shape
    if t <= k:
        raise ValueError("need more observations than members")
    cov = np.cov(e, rowvar=False)
    cov = cov + ridge * np.eye(k)
    ones = np.ones(k, dtype=np.float64)
    sol = np.linalg.solve(cov, ones)
    denom = float(ones @ sol)
    if abs(denom) < 1e-12:
        return np.full(k, 1.0 / k)
    return np.asarray(sol / denom, dtype=np.float64)


def shrink_to_equal(w: FloatArray, gamma: float) -> FloatArray:
    """Shrink combination weights toward equal weights: γ·w + (1−γ)·1/K."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("w must be a non-empty one-dimensional array")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    k = len(w)
    return np.asarray(gamma * w + (1.0 - gamma) * (1.0 / k), dtype=np.float64)


def crps_weights(scores: FloatArray, temperature: float) -> FloatArray:
    """Softmax(−score / T) over members; lower score (CRPS/QLIKE) → more mass.

    ``scores`` is (K,) mean member scores on the calibration window.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = -s / temperature
    z = z - float(np.max(z))
    p = np.exp(z)
    return np.asarray(p / float(p.sum()), dtype=np.float64)
