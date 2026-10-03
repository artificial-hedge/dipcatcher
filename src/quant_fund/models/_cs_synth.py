"""Shared fixture for wave-183 causal-structure-DL canon.

Linear SEM on a random DAG: X = W^T X + noise (upper-triangular W under
random permutation). Ground-truth DAG + SHD metric; baseline = sort by
|correlation| (guessing parents by association strength).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sem_data(seed: int, n: int = 1500, d: int = 6, edges: int = 7) -> tuple[FloatArray, FloatArray]:
    """Returns (X, B_true) with B_true[i,j]=w if i→j edge."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    B = np.zeros((d, d))
    ecount = 0
    while ecount < edges:
        i, j = rng.integers(0, d, 2)
        if i != j and B[i, j] == 0 and B[j, i] == 0:
            B[i, j] = rng.uniform(0.6, 1.2) * rng.choice([-1.0, 1.0])
            # ensure DAG under perm ordering: require perm_rank[i] < perm_rank[j]
            if np.argwhere(perm == i) < np.argwhere(perm == j):
                ecount += 1
            else:
                B[i, j] = 0
    # sample in topological order via B ordering
    X = np.zeros((n, d))
    order = np.argsort([np.argwhere(perm == i)[0, 0] for i in range(d)])
    for j in order:
        X[:, j] = X @ B[:, j] + rng.standard_normal(n) * 0.7
    return X, B


def shd(B_true: FloatArray, B_hat: FloatArray) -> int:
    """Structural Hamming distance (skeleton + orientation errors)."""
    t = np.abs(B_true) > 0.05
    h = np.abs(B_hat) > 0.05
    d = int((t != h).sum() + ((t & h) & (B_true * B_hat < 0)).sum())
    return d // 1


def corr_baseline(X: FloatArray, edges: int) -> FloatArray:
    C: FloatArray = np.asarray(np.corrcoef(X.T))
    np.fill_diagonal(C, 0)
    idx = np.argsort(np.abs(C).ravel())[::-1][: edges * 2]
    B = np.zeros_like(C)
    for k in idx:
        i, j = np.unravel_index(k, C.shape)
        if i < j:
            B[i, j] = C[i, j]
    return B
