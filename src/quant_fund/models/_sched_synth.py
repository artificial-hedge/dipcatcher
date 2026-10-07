"""Shared fixture for wave-198 scheduling canon — flow-shop instances, (SYNTHETIC)
machine loads, knapsack items, TSP distance matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def flowshop(seed: int, n_jobs: int = 12, n_mach: int = 2) -> FloatArray:
    if n_jobs < 1 or n_mach < 1:
        raise ValueError(f"need n_jobs>=1 and n_mach>=1, got {n_jobs},{n_mach}")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(2, 25, (n_jobs, n_mach)).astype(np.float64))


def weighted_jobs(seed: int, n: int = 30) -> tuple[FloatArray, FloatArray]:
    if n < 1:
        raise ValueError(f"need n>=1, got {n}")
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 20, n).astype(np.float64)
    w = rng.integers(1, 10, n).astype(np.float64)
    return p, w


def knapsack(seed: int, n: int = 40, cap_frac: float = 0.4) -> tuple[FloatArray, FloatArray, float]:
    if n < 1 or not 0.0 < cap_frac < 1.0:
        raise ValueError(f"need n>=1 and 0<cap_frac<1, got {n},{cap_frac}")
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 30, n).astype(np.float64)
    v = rng.integers(1, 50, n).astype(np.float64)
    cap = float(w.sum() * cap_frac)
    return w, v, cap


def tsp(seed: int, n: int = 12) -> FloatArray:
    if n < 2:
        raise ValueError(f"need n>=2 cities, got {n}")
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n, 2))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return np.asarray(d)


def makespan(order: list[int], p: np.ndarray) -> float:
    """Flow-shop makespan for a job order."""
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError(f"p must be a non-empty (n_jobs, n_mach) matrix, got {p.shape}")
    if sorted(order) != list(range(p.shape[0])):
        raise ValueError(
            f"order must be a permutation of all jobs; got {sorted(order)} for {p.shape[0]} jobs"
        )
    m = p.shape[1]
    C = np.zeros(m)
    for j in order:
        C[0] += p[j, 0]
        for k in range(1, m):
            C[k] = max(C[k], C[k - 1]) + p[j, k]
    return float(C[-1])
