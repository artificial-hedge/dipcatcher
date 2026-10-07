"""Shared synthetic attributed-graph fixture for wave-135 GNN benches.

Stochastic-block communities + noisy node features: the graph carries signal
the features alone cannot (labels = community; features = centroid + heavy
noise). Every wave-135 bench compares its propagation rule against a
graph-blind MLP on the same features. SYNTHETIC only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def synth_sbm_graph(
    n: int = 200,
    k: int = 4,
    d_feat: int = 16,
    p_in: float = 0.12,
    p_out: float = 0.015,
    feat_noise: float = 1.2,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Returns (adj (n,n), x (n,d), y (n,))."""
    if n < 1 or not 0.0 <= p_in <= 1.0 or not 0.0 <= p_out <= 1.0 or feat_noise < 0.0:
        raise ValueError(
            f"require n>=1, 0<=p_in<=1, 0<=p_out<=1, feat_noise>=0; got n={n}, p_in={p_in}, p_out={p_out}, feat_noise={feat_noise}"
        )
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n).astype(np.int64)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.random() < p:
                adj[i, j] = adj[j, i] = 1.0
    cent = rng.normal(0, 1, (k, d_feat))
    x = cent[y] + feat_noise * rng.standard_normal((n, d_feat))
    return adj.astype(np.float64), x.astype(np.float64), y


def normalize_adj(adj: FloatArray, self_loops: bool = True) -> FloatArray:
    """Symmetric-normalized adjacency D^{-1/2} (A+I) D^{-1/2}."""
    if (np.asarray(adj) < 0).any():
        raise ValueError("adjacency must be non-negative for degree normalization")
    a = adj + (np.eye(adj.shape[0]) if self_loops else 0.0)
    deg = a.sum(1)
    dinv = np.power(np.maximum(deg, 1e-9), -0.5)
    return (dinv[:, None] * a * dinv[None, :]).astype(np.float64)


def split_masks(
    n: int, frac_train: float, seed: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    if n < 1 or not 0.0 <= frac_train <= 1.0:
        raise ValueError(f"require n>=1 and 0<=frac_train<=1, got n={n}, frac_train={frac_train}")
    rng = np.random.default_rng(seed + 777)
    perm = rng.permutation(n)
    n_tr = int(n * frac_train)
    tr = np.zeros(n, dtype=bool)
    tr[perm[:n_tr]] = True
    return tr, ~tr
