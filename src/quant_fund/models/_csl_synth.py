"""Shared fixture for wave-190 classical causal-discovery canon (SYNTHETIC).

SEM generation with selectable noise family (LiNGAM methods need
non-Gaussian noise to identify the order) + induced-skeleton and
ordering-error metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lingam_sem(
    seed: int,
    n: int = 1500,
    d: int = 6,
    edges: int = 7,
    noise: str = "uniform",
) -> tuple[FloatArray, FloatArray, list[int]]:
    """Random DAG SEM with selectable noise. Returns (X, B, order)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d).tolist()
    B = np.zeros((d, d))
    pairs = [(i, j) for i in range(d) for j in range(d) if order.index(i) < order.index(j)]
    sel = rng.choice(len(pairs), size=min(edges, len(pairs)), replace=False)
    for k in sel:
        i, j = pairs[k]
        B[i, j] = rng.uniform(0.6, 1.2) * rng.choice([-1, 1])
    X = np.zeros((n, d))
    for j in order:
        if noise == "uniform":
            e = rng.uniform(-1.2, 1.2, n)
        elif noise == "laplace":
            e = rng.laplace(0, 0.7, n)
        elif noise == "student":
            e = rng.standard_t(3, n) * 0.7
        else:
            e = rng.standard_normal(n) * 0.7
        X[:, j] = X @ B[:, j] + e
    return np.asarray(X), np.asarray(B), order


def skeleton(B: FloatArray) -> FloatArray:
    return np.asarray((B != 0) | (B.T != 0), dtype=np.float64)


def skeleton_f1(B_true: FloatArray, B_hat: FloatArray) -> float:
    sk_t = skeleton(B_true)
    sk_h = skeleton(B_hat)
    tp = float((sk_t * sk_h).sum())
    fp = float(((1 - sk_t) * sk_h).sum())
    fn = float((sk_t * (1 - sk_h)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def order_err(order_true: list[int], order_hat: list[int]) -> float:
    """Kendall-tau style fraction of discordant pairs."""
    pos_t = {v: i for i, v in enumerate(order_true)}
    pos_h = {v: i for i, v in enumerate(order_hat)}
    dis, tot = 0, 0
    for i in range(len(order_true)):
        for j in range(i + 1, len(order_true)):
            a, b = order_true[i], order_true[j]
            tot += 1
            if (pos_h[a] - pos_h[b]) * (pos_t[a] - pos_t[b]) < 0:
                dis += 1
    return dis / max(tot, 1)


def corr_baseline_order(X: FloatArray) -> list[int]:
    """Order variables by total |correlation| (bad heuristic baseline)."""
    C = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(C, 0)
    return list(np.argsort(-C.sum(1)).tolist())
