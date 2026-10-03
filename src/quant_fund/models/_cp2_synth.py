"""Shared SYNTHETIC fixture for wave-171 conformal-2 canon:
heteroscedastic regression + binary classifier. Metrics: marginal
coverage vs 1-alpha target and mean interval width vs split conformal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cp2_data(seed: int, n: int = 600) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (2 * n, 3))
    scale = 0.2 + 0.6 * np.abs(X[:, 0])
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] + scale * rng.standard_normal(2 * n)
    return X[:n], y[:n], X[n:], y[n:]


def fit_ridge(X: FloatArray, y: FloatArray, lam: float = 0.1) -> FloatArray:
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def ridge_pred(w: FloatArray, X: FloatArray) -> FloatArray:
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def cov_width(lo: FloatArray, hi: FloatArray, y: FloatArray) -> tuple[float, float]:
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def cls_data(
    seed: int, n: int = 600
) -> tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (2 * n, 4))
    p = 1 / (1 + np.exp(-(X[:, 0] + 0.7 * X[:, 1] - 0.5)))
    y = (rng.random(2 * n) < p).astype(np.int64)
    return X[:n], y[:n], X[n:], y[n:]
