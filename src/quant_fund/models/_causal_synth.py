"""Causal-inference fixture: confounded treatment with known ITE (SYNTHETIC).

x ~ N(0, I_5); propensity sigmoid(x0); t ~ Bern(e);
y0 = x0 + 0.5*x1 + noise; tau(x) = 1 + x0 (heterogeneous);
y = y0 + tau*t. True ATE = 1, PEHE computable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def synth_observational(
    seed: int = 0, n: int = 800
) -> tuple[FloatArray, IntArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 5))
    e = 1 / (1 + np.exp(-(x[:, 0] + 0.5 * x[:, 2])))
    t = (rng.uniform(0, 1, n) < e).astype(np.int64)
    tau = 1 + x[:, 0]
    y = x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2] + tau * t + rng.normal(0, 0.5, n)
    return x, t, y, tau, e


def synth_iv(seed: int = 0, n: int = 800) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """z→t (instrument), unobserved confounder u affects t,y."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    t = 0.8 * z + 0.7 * u + rng.normal(0, 0.4, n)
    y = 1.5 * t + 0.9 * u + rng.normal(0, 0.4, n)
    return z, t, y, np.full(n, 1.5)
