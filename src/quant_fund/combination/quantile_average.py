"""Quantile averaging (vincentization) for ensemble quantile forecasts.

Quantile averaging pools members per quantile level; for Gaussian members
it is exact and linear in the z-score, and it is *narrower* than probability
averaging (the mixture) — the classic vincentization sharpening result. The
module provides the general weighted quantile average (with monotonicity
repair), closed-form Gaussian variants, and a pinball helper.

Honesty: scores reported are pinball/CRPS on the supplied evaluation set.

References:
- Vincent, S. B. (1912). The function of the vibrissae in the behaviour of
  the white rat — vincentization origin.
- Hora, S. C., Fransen, B. R., Hawkins, N., Susel, I. (2013). Median
  aggregation of distribution forecasts — quantile averaging vs probability
  averaging.
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules —
  CRPS/pinball foundations.

Composition: numpy + scipy.stats; evaluation via quant_fund.metrics.scoring
is done by callers/tests, keeping this module focused on combination.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def quantile_average(member_quantiles: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted average of member quantile forecasts.

    ``member_quantiles`` is (M, ..., Q) with the quantile axis last;
    ``weights`` is (M,) and normalised internally. Output is repaired to be
    monotone non-decreasing across the quantile axis by rearrangement
    (sorting along the last axis), the same operation as
    ``quant_fund.metrics.scoring.rearrange_quantiles``.
    """
    q = np.asarray(member_quantiles, dtype=np.float64)
    if q.ndim < 2:
        raise ValueError("member_quantiles must have a member axis and a quantile axis")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (q.shape[0],):
        raise ValueError("weights must be (M,)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = w / total
    out = np.tensordot(w, q, axes=([0], [0]))
    return np.asarray(np.sort(out, axis=-1), dtype=np.float64)


def quantile_average_gaussian(
    mus: FloatArray,
    sigmas: FloatArray,
    weights: FloatArray,
    taus: FloatArray,
) -> FloatArray:
    """Exact quantile average of Gaussian members: μ̄ + z_τ · Σ w_i σ_i.

    For Gaussians, averaging quantiles is linear, so the result is Gaussian
    with mean Σ w_i μ_i and *arithmetic* mean of σ — hence narrower than the
    probability average, whose dispersion is the RMS √(Σ w_i σ_i²) ≥ Σ w_i σ_i.
    """
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if mus.shape != sigmas.shape or mus.ndim != 1:
        raise ValueError("mus and sigmas must be one-dimensional arrays of equal shape")
    if np.any(sigmas <= 0):
        raise ValueError("sigmas must be positive")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != mus.shape:
        raise ValueError("weights must match mus/sigmas")
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative and sum positive")
    w = w / float(w.sum())
    mu_bar = float(np.dot(w, mus))
    sigma_bar = float(np.dot(w, sigmas))
    z = norm.ppf(np.clip(taus, 1e-9, 1.0 - 1e-9))
    return np.asarray(mu_bar + sigma_bar * z, dtype=np.float64)


def probability_average_gaussian(
    mus: FloatArray,
    sigmas: FloatArray,
    weights: FloatArray,
    taus: FloatArray,
) -> FloatArray:
    """Probability average (mixture) of Gaussians: μ̄ + z_τ · √(Σ w_i σ_i²)."""
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if mus.shape != sigmas.shape or mus.ndim != 1:
        raise ValueError("mus and sigmas must be one-dimensional arrays of equal shape")
    if np.any(sigmas <= 0):
        raise ValueError("sigmas must be positive")
    w = np.asarray(weights, dtype=np.float64)
    w = w / float(w.sum())
    mu_bar = float(np.dot(w, mus))
    var_bar = float(np.dot(w, sigmas * sigmas))
    z = norm.ppf(np.clip(taus, 1e-9, 1.0 - 1e-9))
    return np.asarray(mu_bar + np.sqrt(var_bar) * z, dtype=np.float64)


def pinball(y: FloatArray, q: FloatArray, taus: FloatArray) -> FloatArray:
    """Pinball loss per quantile level (mean over observations)."""
    y = np.asarray(y, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if q.shape[-1] != len(taus):
        raise ValueError("last axis of q must match taus")
    diff = y[..., None] - q
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)
    return np.asarray(np.mean(loss, axis=tuple(range(loss.ndim - 1))), dtype=np.float64)
