"""Shared SYNTHETIC fixture for wave-172 bandit-exotics canon:
K-arm Gaussian bandit + Markov restless arms. Metric: cumulative
regret vs round-robin.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bandit_env(seed: int, K: int = 6) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 1.0, K)
    sd = np.full(K, 0.3)
    return mu, sd


def psrl_env(seed: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Tiny 4-state, 2-action MDP with stochastic rewards/transitions."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(4), (2, 4))  # P[a, s, s']
    R = rng.uniform(0, 1, (2, 4))
    return P[0], P[1], R[0], R[1]
