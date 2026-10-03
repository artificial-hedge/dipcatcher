"""Shared fixture for wave-195 ensemble/adaptive-MCMC canon —
correlated 2-D Gaussian target; metrics = ESS + mean/cov error vs
independent-MH baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MU = np.array([1.0, -0.5])
SIGMA = np.array([[1.0, 0.85], [0.85, 1.0]])
PREC = np.linalg.inv(SIGMA)


def logp(x: FloatArray) -> float:
    d = x - MU
    return float(-0.5 * d @ PREC @ d)


def ess(x: FloatArray, max_lag: int = 50) -> float:
    """Effective sample size via autocorrelation cutoff."""
    n = len(x)
    x = x - x.mean()
    var = float((x**2).mean())
    if var < 1e-12:
        return float(n)
    ess_v = float(n)
    for lag in range(1, min(max_lag, n - 1)):
        rho = float((x[:-lag] * x[lag:]).mean() / var)
        if rho < 0.02:
            break
        ess_v = float(n) / (1 + 2 * rho * lag / lag)
    return float(min(ess_v, n))


def errors(samples: FloatArray) -> tuple[float, float]:
    mean_err = float(np.linalg.norm(samples.mean(0) - MU))
    cov_err = float(np.linalg.norm(np.cov(samples.T) - SIGMA))
    return mean_err, cov_err


def indep_mh(n: int, rng, prop_sd: float = 1.5) -> FloatArray:
    x = np.zeros(2)
    out = []
    for _ in range(n):
        y = x + prop_sd * rng.standard_normal(2)
        if np.log(rng.uniform()) < logp(y) - logp(x):
            x = y
        out.append(x.copy())
    return np.asarray(out)
