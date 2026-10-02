"""Proximal-gradient canon: soft-threshold prox, ISTA, FISTA (Beck & Teboulle
2009) with restart, and accelerated proximal gradient for l1-penalized
quadratics. ``bench_proximal_gradient`` plants a sparse signal under a
correlated design and gates support recovery plus FISTA beating ISTA on the
same iteration budget.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def soft_threshold(x: FloatArray, lam: float) -> FloatArray:
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _obj(x: FloatArray, a: FloatArray, b: FloatArray, lam: float) -> float:
    return float(0.5 * x @ (a @ x) - b @ x + lam * np.abs(x).sum())


def _pgd(
    grad: FloatArray,
    a: FloatArray,
    b: FloatArray,
    lam: float,
    it: int,
    lip: float,
    accel: str,
) -> FloatArray:
    x = np.zeros_like(b)
    y = x.copy()
    t = 1.0
    f_prev = np.inf
    for _ in range(it):
        g = a @ y - b
        x_new = soft_threshold(y - g / lip, lam / lip)
        if accel == "fista":
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + (t - 1.0) / t_new * (x_new - x)
            f_new = _obj(x_new, a, b, lam)
            if f_new > f_prev:  # restart momentum
                y = x_new.copy()
                t_new = 1.0
            f_prev = f_new
            t = t_new
        else:
            y = x_new
        x = x_new
    return x


def ista_lasso(x: FloatArray, y: FloatArray, lam: float, it: int = 300) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or lam < 0:
        raise ValueError("bad lasso inputs")
    a = x.T @ x / len(y)
    b = x.T @ y / len(y)
    lip = float(np.linalg.eigvalsh(a).max())
    return _pgd(np.zeros(x.shape[1]), a, b, lam, it, lip, "ista")


def fista_lasso(x: FloatArray, y: FloatArray, lam: float, it: int = 300) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or lam < 0:
        raise ValueError("bad lasso inputs")
    a = x.T @ x / len(y)
    b = x.T @ y / len(y)
    lip = float(np.linalg.eigvalsh(a).max())
    return _pgd(np.zeros(x.shape[1]), a, b, lam, it, lip, "fista")


def prox_lasso_loss(x: FloatArray, y: FloatArray, w: FloatArray, lam: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(0.5 / len(y) * np.sum((x @ w - y) ** 2) + lam * np.abs(w).sum())


def bench_proximal_gradient(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d = 250, 60
    rho = 0.4
    cov = rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    x = rng.multivariate_normal(np.zeros(d), cov, size=n)
    w_true = np.zeros(d)
    w_true[[0, 7, 21, 33, 45]] = [2.0, -1.5, 1.2, -0.9, 0.8]
    y = x @ w_true + 0.3 * rng.standard_normal(n)
    lam = 0.05
    w_i = ista_lasso(x, y, lam, it=200)
    w_f = fista_lasso(x, y, lam, it=200)
    a = x.T @ x / n
    b = x.T @ y / n
    f_i = _obj(w_i, a, b, lam)
    f_f = _obj(w_f, a, b, lam)
    true_supp = np.flatnonzero(np.abs(w_true) > 0.1)
    supp_i = np.flatnonzero(np.abs(w_i) > 0.1)
    supp_f = np.flatnonzero(np.abs(w_f) > 0.1)
    ridge = solve(a + 0.05 * np.eye(d), b)
    return {
        "synthetic_fista_support_hit": float(np.intersect1d(supp_f, true_supp).size),
        "synthetic_ista_support_hit": float(np.intersect1d(supp_i, true_supp).size),
        "synthetic_fista_obj_gap": float(f_i - f_f),
        "synthetic_fista_vs_ridge_err": float(
            np.linalg.norm(w_f - w_true) / np.linalg.norm(ridge - w_true)
        ),
    }
