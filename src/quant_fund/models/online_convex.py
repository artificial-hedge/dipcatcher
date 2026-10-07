"""Online convex learning: classic perceptron, (SYNTHETIC)
passive-aggressive (PA-I, Crammer et al. 2006), online
gradient descent on logistic loss, and follow-the-
regularized-leader (FTRL-Proximal, McMahan 2011). Synthetic
bench gates online accuracy against a batch baseline on a
noisy linear fixture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def perceptron_online(x: FloatArray, y: FloatArray, it: int = 3, seed: int = 0) -> FloatArray:
    """w ← w + y_i x_i on misclassification, it epochs."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64) * 2 - 1
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    for _ in range(it):
        for i in rng.permutation(len(x)):
            if y[i] * float(x[i] @ w) <= 0:
                w += y[i] * x[i]
    return np.asarray(w)


def passive_aggressive(
    x: FloatArray, y: FloatArray, c: float = 0.5, it: int = 3, seed: int = 0
) -> FloatArray:
    """PA-I: τ = min(C, max(0, 1 − ywx) / ‖x‖²) update."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64) * 2 - 1
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    for _ in range(it):
        for i in rng.permutation(len(x)):
            loss = 1.0 - y[i] * float(x[i] @ w)
            if loss > 0:
                tau = min(c, loss / float(x[i] @ x[i] + 1e-12))
                w += tau * y[i] * x[i]
    return np.asarray(w)


def ogd_logistic(
    x: FloatArray, y: FloatArray, lr: float = 0.3, it: int = 2, seed: int = 0
) -> FloatArray:
    """w ← w − lr·∇logistic."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1])
    for _ in range(it):
        for i in rng.permutation(len(x)):
            p = 1.0 / (1.0 + np.exp(-np.clip(float(x[i] @ w), -30, 30)))
            w -= lr * (p - y[i]) * x[i]
    return np.asarray(w)


def ftrl_proximal(
    x: FloatArray,
    y: FloatArray,
    alpha: float = 0.3,
    beta: float = 1.0,
    l1: float = 0.01,
    l2: float = 0.01,
    seed: int = 0,
) -> FloatArray:
    """FTRL-proximal with per-coordinate step sizes."""
    x = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = np.zeros(x.shape[1])
    z = np.zeros(x.shape[1])
    w = np.zeros(x.shape[1])
    for i in rng.permutation(len(x)):
        xi, yi = x[i], y[i]
        p = 1.0 / (1.0 + np.exp(-np.clip(float(xi @ w), -30, 30)))
        g = (p - yi) * xi
        sigma = (np.sqrt(n + g**2) - np.sqrt(n)) / alpha
        z += g - sigma * w
        n += g**2
        w = np.where(
            np.abs(z) > l1,
            -(z - np.sign(z) * l1) / ((beta + np.sqrt(n)) / alpha + l2),
            0.0,
        )
    return np.asarray(w)


def _pred(w: FloatArray, x: FloatArray) -> FloatArray:
    xa = np.c_[np.asarray(x, dtype=np.float64), np.ones(len(x))]
    if w.shape[0] == xa.shape[1]:
        return np.asarray((xa @ w > 0).astype(np.float64))
    return np.asarray((np.asarray(x) @ w > 0).astype(np.float64))


def bench_online_convex(seed: int = 557) -> dict[str, float]:
    """SYNTHETIC: noisy linearly separable fixture — every
    online learner must reach batch-comparable accuracy and
    beat a mean-guess baseline."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 300
    x = rng.normal(0, 1, (n, 4))
    beta = np.array([1.5, -1.0, 0.8, 0.4])
    y = ((x @ beta + rng.normal(0, 0.8, n)) > 0).astype(np.float64)
    perm = rng.permutation(n)
    tr, te = perm[:220], perm[220:]
    accs = {}
    accs["perceptron"] = float(
        (_pred(perceptron_online(x[tr], y[tr], it=4, seed=seed), x[te]) == y[te]).mean()
    )
    accs["pa"] = float(
        (_pred(passive_aggressive(x[tr], y[tr], c=0.5, it=4, seed=seed), x[te]) == y[te]).mean()
    )
    accs["ogd"] = float(
        (_pred(ogd_logistic(x[tr], y[tr], lr=0.3, it=3, seed=seed), x[te]) == y[te]).mean()
    )
    accs["ftrl"] = float((_pred(ftrl_proximal(x[tr], y[tr], seed=seed), x[te]) == y[te]).mean())
    for name, a in accs.items():
        out[f"synthetic_{name}_acc"] = a
        if a < 0.75:
            raise ValueError(f"{name} acc off: {a}")
    return out
