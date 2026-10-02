"""Shared fixture for wave-192 info-theory canon — dependent/independent
data pairs for two-sample and independence testing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def dep_data(seed: int, n: int = 600, kind: str = "dep") -> tuple[FloatArray, FloatArray]:
    """kind: 'dep' (x→y strongly), 'indep', 'nonlin' (xor/circle dependence)."""
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
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = rng.standard_normal((n, 2)) + shift
    return np.asarray(x), np.asarray(y)


def rbf(x: FloatArray, y: FloatArray, bw: float) -> FloatArray:
    d2 = x[:, None, :] - y[None, :, :]
    return np.exp(-(d2**2).sum(-1) / (2 * bw**2))


def med_bw(X: FloatArray, Y: FloatArray) -> float:
    Z = np.vstack([X, Y])
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return float(np.asarray(np.median(d[d > 0])))


def true_mi_gauss(corr: float) -> float:
    return float(-0.5 * np.log(max(1 - corr**2, 1e-9)))
