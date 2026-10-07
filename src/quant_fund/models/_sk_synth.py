"""Shared SYNTHETIC fixture for wave-177 neuromorphic canon:
binary task Poisson-encoded into spike trains (T=30 steps, d=8
inputs); LIF dynamics + rate/latency decoding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sk_data(seed: int, n: int = 300, d: int = 8) -> tuple[FloatArray, NDArray[np.int64]]:
    if n < 1 or d < 1:
        raise ValueError(f"need n,d >= 1, got {n},{d}")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    y = (X @ w * 2 + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def poisson_encode(X: FloatArray, T: int = 30, seed: int = 0) -> FloatArray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"X must be a non-empty 2-D array, got {X.shape}")
    if T < 1:
        raise ValueError(f"need T>=1 spike steps, got {T}")
    rng = np.random.default_rng(seed)
    rates = 1.0 / (1.0 + np.exp(-X))  # rates in (0,1)
    return (rng.random((len(X), X.shape[1], T)) < rates[:, :, None] / 3).astype(np.float64)


def lif_forward(spikes: FloatArray, w: FloatArray, T: int = 30) -> FloatArray:
    """LIF membrane trace: v_t = a v_{t-1} + w.s_t; returns final v."""
    spikes = np.asarray(spikes, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if spikes.ndim != 3 or spikes.shape[2] < T or T < 1:
        raise ValueError(f"need spikes (n, d, >=T) with T>=1, got shape {spikes.shape}, T={T}")
    if w.shape != (spikes.shape[1],):
        raise ValueError(f"w must be (d={spikes.shape[1]},), got {w.shape}")
    a = 0.85
    v = np.zeros(len(spikes))
    for t in range(T):
        v = a * v + spikes[:, :, t] @ w
    return v


def latency_encode(X: FloatArray, T: int = 30) -> FloatArray:
    """First-spike latency: high input → early spike (one-hot over T)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"X must be a non-empty 2-D array, got {X.shape}")
    if T < 1:
        raise ValueError(f"need T>=1 spike steps, got {T}")
    rates = 1.0 / (1.0 + np.exp(-X))
    lat = np.clip(((1 - rates) * T).astype(int), 0, T - 1)
    sp = np.zeros((len(X), X.shape[1], T))
    for i in range(len(X)):
        for j in range(X.shape[1]):
            sp[i, j, lat[i, j]] = 1.0
    return sp
