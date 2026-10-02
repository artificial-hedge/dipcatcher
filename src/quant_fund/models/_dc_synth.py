"""Shared SYNTHETIC fixture for wave-176 data-centric canon:
2-class task — clean cluster + 20% label-flipped/hard region; a
labeled pool + oracle that returns true labels on query. Metric:
post-budget accuracy vs random selection / full-data upper bound.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def dc_data(
    seed: int, n: int = 400, d: int = 6, flip: float = 0.15
) -> tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    """Pool X with noisy labels, test set clean."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    p = 1.0 / (1.0 + np.exp(-(X @ w) * 3))
    y = (rng.random(n) < p).astype(np.int64)
    fl = rng.random(n) < flip
    y[fl] = 1 - y[fl]
    Xt = rng.standard_normal((300, d))
    yt = (Xt @ w * 3 + 0.5 * rng.standard_normal(300) > 0).astype(np.int64)
    return X, y, Xt, yt


def fit_eval(X: FloatArray, y: NDArray[np.int64], Xt: FloatArray, yt: NDArray[np.int64]) -> float:
    """Logistic-regression accuracy."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.zeros(Xb.shape[1])
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y) / len(X) + 0.001 * w
        w -= 0.5 * g
    pb = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)
    return float(((pb @ w > 0).astype(np.int64) == yt).mean())
