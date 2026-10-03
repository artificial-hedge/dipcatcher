"""Shared fixture for wave-198 scheduling canon — flow-shop instances,
machine loads, knapsack items, TSP distance matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def flowshop(seed: int, n_jobs: int = 12, n_mach: int = 2) -> FloatArray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(2, 25, (n_jobs, n_mach)).astype(np.float64))


def weighted_jobs(seed: int, n: int = 30) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 20, n).astype(np.float64)
    w = rng.integers(1, 10, n).astype(np.float64)
    return p, w


def knapsack(seed: int, n: int = 40, cap_frac: float = 0.4) -> tuple[FloatArray, FloatArray, float]:
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 30, n).astype(np.float64)
    v = rng.integers(1, 50, n).astype(np.float64)
    cap = float(w.sum() * cap_frac)
    return w, v, cap


def tsp(seed: int, n: int = 12) -> FloatArray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n, 2))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return np.asarray(d)


def makespan(order: list[int], p: np.ndarray) -> float:
    """Flow-shop makespan for a job order."""
    m = p.shape[1]
    C = np.zeros(m)
    for j in order:
        C[0] += p[j, 0]
        for k in range(1, m):
            C[k] = max(C[k], C[k - 1]) + p[j, k]
    return float(C[-1])
