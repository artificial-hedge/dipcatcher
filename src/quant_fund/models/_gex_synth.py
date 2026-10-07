"""Shared graph-exotics fixture: planted-clique node classification on (SYNTHETIC)
random graphs (n=16 nodes, one planted 5-clique + background Erdos–Renyi).
Metric: node AUC vs feature-only MLP baseline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def planted_clique(
    seed: int = 7, n: int = 16, k: int = 5, p: float = 0.15
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (adjacency, features, labels). Features: degree + random."""
    if not 0.0 <= p <= 1.0 or k < 2 or n < 1:
        raise ValueError(f"require 0<=p<=1, k>=2, n>=1; got p={p}, k={k}, n={n}")
    rng = np.random.default_rng(seed)
    A = (rng.uniform(size=(n, n)) < p).astype(np.float64)
    A = np.maximum(A, A.T)  # undirected Erdos–Renyi
    np.fill_diagonal(A, 0)
    clique = rng.choice(n, k, replace=False)
    for i in clique:
        for j in clique:
            if i != j:
                A[i, j] = A[j, i] = 1.0
    deg = A.sum(1, keepdims=True)
    x = np.concatenate([deg / n, rng.standard_normal((n, 3)) * 0.3], 1)
    y = np.zeros(n)
    y[clique] = 1.0
    return A, x.astype(np.float64), y.astype(np.float64)
