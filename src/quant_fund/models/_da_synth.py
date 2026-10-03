"""Shared fixture for wave-185 differentiable-algorithm canon.

Top-k selection task: learn scores s (d=6) so that weighted sum of the
top-2 selected features hits a target. Selection is discrete — the
forward is non-differentiable; each method estimates a usable gradient.
Ground truth: central finite differences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

K = 2
W = np.array([1.5, -0.5, 1.0, 0.3, -1.2, 0.8])
TARGET = 2.0


def sel_data(seed: int = 0) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    s0 = rng.standard_normal(6) * 0.3
    return s0, rng.standard_normal(6)


def topk_mask(s: FloatArray, k: int = K) -> FloatArray:
    m = np.zeros_like(s)
    m[np.argsort(-s)[:k]] = 1.0
    return m


def loss_np(s: FloatArray) -> float:
    return float((W @ (topk_mask(s) * s) - TARGET) ** 2)


def finite_diff_grad(s: FloatArray, eps: float = 1e-4) -> FloatArray:
    g = np.zeros_like(s)
    for i in range(len(s)):
        d = np.zeros_like(s)
        d[i] = eps
        g[i] = (loss_np(s + d) - loss_np(s - d)) / (2 * eps)
    return g


def grad_corr(a: FloatArray, b: FloatArray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))
