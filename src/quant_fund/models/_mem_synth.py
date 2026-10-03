"""Shared synthetic fixtures for the wave-138 memory + world-model canon.

- `synth_copy`: read a K-symbol one-hot sequence, then emit it — memory
  mechanisms (external store, kNN over past keys, LSH buckets) are what
  solve it where local context windows fail.
- `synth_dynamics`: noisy 2-D controlled oscillator for latent-dynamics
  prediction and shooting-based planning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_K = 8


def synth_copy(n: int, t: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    """x: (n, 2t+1, K) — first t tokens are the payload, a separator token
    (all-zero + delim channel is implicit: we use a dedicated flag dim),
    then t blank slots the model must fill with the payload again."""
    k = _K + 1
    x = np.zeros((n, 2 * t + 1, k))
    y = np.zeros((n, t), dtype=np.int64)
    payload = rng.integers(0, _K, (n, t))
    x[:, :t, :_K] = np.eye(_K)[payload]
    x[:, t, _K] = 1.0
    y = payload
    return x, y


def synth_dynamics(
    n: int, horizon: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """x_{t+1} = A x_t + b u_t + eps; returns (x_seq, u_seq, x_next_seq)."""
    a = np.array([[0.9, -0.4], [0.4, 0.9]])
    b = np.array([0.3, 0.15])
    x = np.zeros((n, horizon + 1, 2))
    u = rng.uniform(-1.0, 1.0, (n, horizon, 1))
    x[:, 0] = rng.normal(0, 0.5, (n, 2))
    for tt in range(horizon):
        x[:, tt + 1] = x[:, tt] @ a.T + u[:, tt] * b + 0.02 * rng.standard_normal((n, 2))
    return x, u, x[:, 1:]
