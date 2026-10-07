"""Shared SYNTHETIC fixture for wave-169 amortized-inference canon:
Bayesian logistic regression (nonconjugate). Oracle posterior via
random-walk Metropolis–Hastings. Metric: held-out log-loss +
deviation of posterior mean from MCMC oracle.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def vi_data(
    seed: int, n: int = 300, d: int = 5
) -> tuple[FloatArray, NDArray[np.int64], FloatArray, NDArray[np.int64], FloatArray]:
    if n < 1 or d < 1:
        raise ValueError(f"need n,d >= 1, got {n},{d}")
    rng = np.random.default_rng(seed)
    w_true = rng.normal(0.0, 1.5, d)
    X = rng.normal(0.0, 1.0, (2 * n, d))
    p = 1.0 / (1.0 + np.exp(-X @ w_true))
    y = (rng.random(2 * n) < p).astype(np.int64)
    return X[:n], y[:n], X[n:], y[n:], w_true


def logpost(w: FloatArray, X: FloatArray, y: NDArray[np.int64], s0: float = 3.0) -> float:
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != w.shape[0]:
        raise ValueError(f"X must be (n,d={w.shape[0]}), got {X.shape}")
    if len(y) != X.shape[0] or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must be a binary vector matching X rows")
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError(f"need finite s0>0, got {s0}")
    eta = np.clip(X @ w, -30, 30)
    ll = float(np.sum(y * (-np.log1p(np.exp(-eta))) + (1 - y) * (-eta - np.log1p(np.exp(-eta)))))
    return ll - 0.5 * float(w @ w) / s0**2


def mcmc_oracle(
    X: FloatArray, y: NDArray[np.int64], seed: int, iters: int = 6000
) -> tuple[FloatArray, FloatArray]:
    if iters < 2:
        raise ValueError(f"need iters>=2, got {iters}")
    if X.ndim != 2 or X.shape[0] < 1 or len(y) != X.shape[0]:
        raise ValueError(f"X must be non-empty (n,d) with y matching, got {X.shape},{len(y)}")
    rng = np.random.default_rng(seed + 999)
    w = np.zeros(X.shape[1])
    lp = logpost(w, X, y)
    acc = []
    for _ in range(iters):
        wp = w + 0.15 * rng.standard_normal(w.size)
        lp2 = logpost(wp, X, y)
        if np.log(rng.random()) < lp2 - lp:
            w, lp = wp, lp2
        acc.append(w.copy())
    A = np.asarray(acc[iters // 2 :])
    return A.mean(0), A.std(0)


def test_logloss(w: FloatArray, Xt: FloatArray, yt: NDArray[np.int64]) -> float:
    if Xt.ndim != 2 or Xt.shape[0] < 1 or Xt.shape[1] != w.shape[0]:
        raise ValueError(f"Xt must be (n,d={w.shape[0]}), got {Xt.shape}")
    if len(yt) != Xt.shape[0] or not set(np.unique(yt)) <= {0, 1}:
        raise ValueError("yt must be a binary vector matching Xt rows")
    p = 1.0 / (1.0 + np.exp(-np.clip(Xt @ w, -30, 30)))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -float(np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p)))
