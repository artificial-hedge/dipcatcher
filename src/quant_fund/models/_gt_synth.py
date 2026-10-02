"""Shared SYNTHETIC fixture for wave-174 graph-temporal canon:
8-node ring topology, node signals are phase-shifted sinusoids +
AR noise so neighbors predict each other. Metric: next-step MSE
vs per-node univariate AR(2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ring_adj(n: int = 8) -> FloatArray:
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i - 1) % n] = A[i, (i + 1) % n] = 1.0
    D = A.sum(1)
    return A / np.maximum(D, 1)[:, None]


def gt_data(seed: int, T: int = 300, n: int = 8) -> FloatArray:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.sin(2 * np.pi * t[:, None] / 40 + ph[None]) + 0.2 * np.sin(
        2 * np.pi * t[:, None] / 11 + ph[None]
    )
    X += 0.15 * rng.standard_normal((T, n))
    # AR smoothing along time
    for i in range(2, T):
        X[i] += 0.4 * X[i - 1]
    return X.astype(np.float64)


def ar2_baseline(X: FloatArray) -> float:
    """Per-node AR(2) one-step MSE."""
    T, n = X.shape
    errs = []
    for k in range(n):
        y = X[3:, k]
        D = np.stack([X[2:-1, k], X[1:-2, k], np.ones(T - 3)], 1)
        w = np.linalg.solve(D.T @ D + 0.01 * np.eye(3), D.T @ y)
        errs.append(np.mean((y - D @ w) ** 2))
    return float(np.mean(errs))
