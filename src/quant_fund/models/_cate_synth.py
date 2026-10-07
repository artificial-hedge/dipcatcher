"""Shared SYNTHETIC fixture for wave-170 causal-DL-2 canon: confounded
treatment with heterogeneous effect tau(x) = 1.2·x0 for x1>0 else -0.4.
Metric: PEHE
(sqrt MSE of estimated CATE) vs naive difference regressor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cate_data(
    seed: int, n: int = 500, d: int = 6
) -> tuple[FloatArray, NDArray[np.int64], FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    # confounded treatment: depends on x0
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.3)))
    t = (rng.random(n) < p).astype(np.int64)
    tau = np.where(X[:, 1] > 0, 1.2 * X[:, 0], -0.4)
    mu0 = np.sin(X[:, 0]) + 0.5 * X[:, 2]
    y = mu0 + t * tau + 0.3 * rng.standard_normal(n)
    return X, t, y, tau, mu0 + tau  # features, treat, obs, true CATE, E[Y(1)]


def pehe(tau_hat: FloatArray, tau: FloatArray) -> float:
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))
