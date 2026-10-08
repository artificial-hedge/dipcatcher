"""Local test helpers: minimal proper-score definitions for this suite.

The combination tests score ensembles against a compact local CRPS/pinball
implementation (Gneiting–Raftery Riemann-sum form), keeping this suite
runnable with numpy alone. Production evaluation should use
``quant_fund.metrics.scoring`` (same definitions, fuller API).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pinball_loss(y: FloatArray, q: FloatArray, tau: float) -> FloatArray:
    diff = y - q
    return np.asarray(np.maximum(tau * diff, (tau - 1.0) * diff), dtype=np.float64)


def crps_from_quantiles(y: FloatArray, quantiles: FloatArray, taus: FloatArray) -> float:
    """Riemann-sum CRPS approximation from pinball losses (Gneiting-Raftery)."""
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    if q.ndim != 2 or q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching taus")
    y = np.asarray(y, dtype=float)
    dt = np.diff(np.concatenate([[0.0], t]))
    total = np.zeros(y.shape[0], dtype=float)
    for k, tau in enumerate(t):
        total += 2.0 * pinball_loss(y, q[:, k], float(tau)) * dt[k]
    return float(np.mean(total))


def crps_gaussian(y: FloatArray, mu: FloatArray, sigma: FloatArray) -> FloatArray:
    """Closed-form CRPS for N(mu, sigma^2) forecasts (Gneiting et al. 2005)."""
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = (y - mu) / sigma
    return np.asarray(
        sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)),
        dtype=np.float64,
    )
