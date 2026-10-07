"""Shared PDE operator-learning fixture: 1-D Poisson -u'' = a(x) on (SYNTHETIC)
[0,1] with Dirichlet BCs, solved by sine transform; pairs (a, u) on a
uniform grid for the neural-operator canon.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def poisson_pairs(
    seed: int = 7,
    n_samp: int = 200,
    n_grid: int = 32,
    k_max: int = 6,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Random forcing a(x) = Σ c_k sin(kπx) → u = Σ c_k sin(kπx)/(kπ)^2."""
    if n_samp < 2 or n_grid < 2 or k_max < 1:
        raise ValueError(f"need n_samp>=2, n_grid>=2, k_max>=1, got {n_samp},{n_grid},{k_max}")
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_grid)
    coeffs = rng.standard_normal((n_samp, k_max)) / (np.arange(k_max) + 1) ** 1.5
    a = np.zeros((n_samp, n_grid))
    u = np.zeros((n_samp, n_grid))
    for k in range(1, k_max + 1):
        a += coeffs[:, k - 1 : k] * np.sin(np.pi * k * x)[None, :]
        u += (coeffs[:, k - 1 : k] / (np.pi * k) ** 2) * np.sin(np.pi * k * x)[None, :]
    n_tr = n_samp // 2
    return (
        a[:n_tr].astype(np.float64),
        u[:n_tr].astype(np.float64),
        a[n_tr:].astype(np.float64),
        u[n_tr:].astype(np.float64),
    )


def rel_l2(pred: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    if pred.ndim != 2 or truth.ndim != 2 or pred.shape != truth.shape:
        raise ValueError(f"need matching (n, g) shapes, got {pred.shape} vs {truth.shape}")
    if pred.shape[0] < 1 or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("need non-empty finite inputs")
    denom = np.linalg.norm(truth, axis=1).mean()
    if denom <= 0:
        raise ValueError("relative L2 undefined for zero-norm truth")
    return float(np.linalg.norm(pred - truth, axis=1).mean() / denom)
