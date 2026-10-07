"""Shared fixture for wave-188 active-learning canon (SYNTHETIC).

Pool-based AL: 2-D binary classification with a curved boundary
y = 1[x2 > sin(2.5x1) + 0.3x1²]; feature map (x1, x2, x1², sinx1, x1x2)
+ logistic regression. Each strategy selects `batch` unlabeled points
per round for `rounds` rounds; accuracy on held-out test tracked vs a
random-selection baseline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def feats(X: FloatArray) -> FloatArray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1, x2, x1 * x1, np.sin(2.5 * x1), x1 * x2, np.ones(len(X))])


def make_data(
    seed: int = 0, n_pool: int = 600, n_test: int = 400
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    Xp = rng.uniform(-2.5, 2.5, (n_pool, 2))
    yp = (Xp[:, 1] > np.sin(2.5 * Xp[:, 0]) + 0.3 * Xp[:, 0] ** 2).astype(np.float64)
    Xt = rng.uniform(-2.5, 2.5, (n_test, 2))
    yt = (Xt[:, 1] > np.sin(2.5 * Xt[:, 0]) + 0.3 * Xt[:, 0] ** 2).astype(np.float64)
    return Xp, yp, Xt, yt


def fit_logreg(P: FloatArray, y: FloatArray, iters: int = 300, lr: float = 0.5) -> FloatArray:
    w = np.zeros(P.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(P @ w, -30, 30)))
        w -= lr * P.T @ (p - y) / len(y)
    return w


def acc(w: FloatArray, Xt: FloatArray, yt: FloatArray) -> float:
    return float(((feats(Xt) @ w > 0) == yt).mean())


def al_loop(
    select: Callable[[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], FloatArray],
    seed: int = 0,
    n_init: int = 10,
    batch: int = 8,
    rounds: int = 10,
) -> float:
    """Run an AL loop; `select(unlabeled_P, unlabeled_idx, labeled_P, w)` →
    batch indices into the unlabeled set. Returns final test accuracy."""
    Xp, yp, Xt, yt = make_data(seed)
    rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(len(Xp))
    labeled = idx[:n_init].tolist()
    unlabeled = set(idx[n_init:].tolist())
    for _ in range(rounds):
        P = feats(Xp[np.array(labeled)])
        w = fit_logreg(P, yp[np.array(labeled)])
        u_idx = np.array(sorted(unlabeled))
        sel = select(
            feats(Xp[u_idx]),
            u_idx,
            feats(Xp[np.array(labeled)]),
            yp[np.array(labeled)],
            w,
        )
        for j in np.atleast_1d(sel)[:batch]:
            unlabeled.discard(int(u_idx[int(j)]))
            labeled.append(int(u_idx[int(j)]))
    w = fit_logreg(feats(Xp[np.array(labeled)]), yp[np.array(labeled)])
    return acc(w, Xt, yt)


def random_baseline(seed: int = 0, rounds: int = 10, batch: int = 8) -> float:
    sel_rng = np.random.default_rng(seed + 101)

    def rnd(
        uP: FloatArray, u_idx: FloatArray, lP: FloatArray, ly: FloatArray, w: FloatArray
    ) -> FloatArray:
        return np.asarray(sel_rng.permutation(len(uP)), dtype=np.float64)

    return al_loop(rnd, seed=seed, batch=batch, rounds=rounds)


def probs(uP: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(1 / (1 + np.exp(-np.clip(uP @ w, -30, 30))))
