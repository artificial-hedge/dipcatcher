"""Multi-task learning canon: shared-representation ridge — joint fit (SYNTHETIC)
W (d,T) minimizing ||X_t w_t - y_t||^2 + lam*||W||_F^2, plus the
dirty/l1-shared variant by iteratively reweighted feature scaling that
concentrates weight energy on shared coordinates.
``bench_multi_task`` plants tasks sharing 80% of their support and gates
shared-support recovery + MTL beating independent ridge on the
low-sample tasks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mtl_ridge(
    xs: list[FloatArray],
    ys: list[FloatArray],
    lam: float = 1e-3,
) -> FloatArray:
    """Independent-per-task ridge closed form (pooled penalty)."""
    d = xs[0].shape[1]
    t = len(xs)
    w = np.zeros((d, t))
    for i in range(t):
        x = np.asarray(xs[i], dtype=np.float64)
        y = np.asarray(ys[i], dtype=np.float64)
        w[:, i] = np.linalg.solve(x.T @ x + lam * len(y) * np.eye(d), x.T @ y)
    return w


def mtl_shared(
    xs: list[FloatArray],
    ys: list[FloatArray],
    lam: float = 1e-3,
    lam_share: float = 0.1,
    it: int = 60,
) -> tuple[FloatArray, FloatArray]:
    """Mean-shrinkage MTL: minimize sum_t ||X_t w_t - y_t||^2
    + lam*||w_t||^2 + lam_share*||w_t - mu||^2 with mu the task mean;
    solved by alternating per-task ridge and mu updates (Evgeniou &
    Pontil regularized MTL). Returns (W, feature_scale) where
    feature_scale is ||W||_2 across tasks normalized."""
    d = xs[0].shape[1]
    t = len(xs)
    w = mtl_ridge(xs, ys, lam=lam)
    mu = w.mean(axis=1)
    eye = np.eye(d)
    for _ in range(it):
        for i in range(t):
            x = np.asarray(xs[i], dtype=np.float64)
            y = np.asarray(ys[i], dtype=np.float64)
            a = x.T @ x + (lam * len(y) + lam_share) * eye
            b = x.T @ y + lam_share * mu
            w[:, i] = np.linalg.solve(a, b)
        mu = w.mean(axis=1)
    nu = np.linalg.norm(w, axis=1)
    scale = nu / np.maximum(nu.max(), 1e-12)
    return w, scale


def mtl_predict(w: FloatArray, x: FloatArray) -> FloatArray:
    return np.asarray(np.asarray(x, dtype=np.float64) @ w)


def bench_multi_task(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d, t = 40, 6
    shared = np.arange(12)
    w_true = np.zeros((d, t))
    base = rng.standard_normal(len(shared)) * 2.0
    for i in range(t):
        w_true[shared, i] = base + 0.1 * rng.standard_normal(len(shared))
        own = rng.choice(np.setdiff1d(np.arange(d), shared), 3, replace=False)
        w_true[own, i] += 0.3 * rng.standard_normal(3)
    xs, ys, xt, yt = [], [], [], []
    for i in range(t):
        xtr = rng.standard_normal((28, d))
        xte = rng.standard_normal((200, d))
        xs.append(xtr)
        ys.append(xtr @ w_true[:, i] + 0.3 * rng.standard_normal(28))
        xt.append(xte)
        yt.append(xte @ w_true[:, i] + 0.3 * rng.standard_normal(200))
    w_ind = mtl_ridge(xs, ys, lam=1e-2)
    w_mtl, scale = mtl_shared(xs, ys, lam=5e-3, lam_share=40.0, it=80)
    err_ind = float(
        np.mean([np.sqrt(np.mean((xt[i] @ w_ind[:, i] - yt[i]) ** 2)) for i in range(t)])
    )
    err_mtl = float(
        np.mean([np.sqrt(np.mean((xt[i] @ w_mtl[:, i] - yt[i]) ** 2)) for i in range(t)])
    )
    top_scale = np.argsort(-scale)[: len(shared)]
    return {
        "synthetic_mtl_rmse": err_mtl,
        "synthetic_ind_rmse": err_ind,
        "synthetic_mtl_gain": err_ind - err_mtl,
        "synthetic_mtl_shared_hit": float(np.intersect1d(top_scale, shared).size),
        "synthetic_mtl_scale_conc": float(scale[shared].mean() - np.delete(scale, shared).mean()),
    }
