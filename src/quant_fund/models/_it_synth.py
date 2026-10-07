"""Shared fixture for wave-192 info-theory canon — dependent/independent (SYNTHETIC)
data pairs for two-sample and independence testing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def dep_data(seed: int, n: int = 600, kind: str = "dep") -> tuple[FloatArray, FloatArray]:
    """kind: 'dep' (x→y strongly), 'indep', 'nonlin' (xor/circle dependence)."""
    if kind not in ("dep", "indep", "nonlin"):
        raise ValueError(f"unknown dependence kind {kind!r}; expected 'dep', 'indep' or 'nonlin'")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    if kind == "dep":
        x = rng.standard_normal(n)
        y = 0.8 * x + 0.2 * rng.standard_normal(n)
    elif kind == "nonlin":
        x = rng.uniform(-np.pi, np.pi, n)
        y = np.sin(2 * x) + 0.1 * rng.standard_normal(n)
    else:
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
    return np.asarray(x), np.asarray(y)


def twosample(seed: int, n: int = 400, shift: float = 0.0) -> tuple[FloatArray, FloatArray]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = rng.standard_normal((n, 2)) + shift
    return np.asarray(x), np.asarray(y)


def rbf(x: FloatArray, y: FloatArray, bw: float) -> FloatArray:
    if not bw > 0.0 or not np.isfinite(bw):
        raise ValueError(f"kernel bandwidth must be positive and finite, got {bw}")
    d2 = x[:, None, :] - y[None, :, :]
    return np.exp(-(d2**2).sum(-1) / (2 * bw**2))


def med_bw(X: FloatArray, Y: FloatArray) -> float:
    Z = np.vstack([X, Y])
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    pos = d[d > 0]
    if pos.size == 0:
        raise ValueError("median heuristic undefined: all pairwise distances are zero")
    return float(np.asarray(np.median(pos)))


def true_mi_gauss(corr: float) -> float:
    if not -1.0 < corr < 1.0:
        raise ValueError(f"Gaussian MI requires |corr| < 1, got {corr}")
    return float(-0.5 * np.log(1 - corr**2))
