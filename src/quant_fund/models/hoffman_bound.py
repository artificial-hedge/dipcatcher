"""Hoffman ratio bound on the independence number of a regular graph.

alpha(G) <= n * (-lambda_min) / (d - lambda_min) for d-regular G.
Bench verifies the bound holds vs the brute-force independence number
on a generated 4-regular graph, and reports the spectral gap inputs.
"""

import itertools

import numpy as np


def _brute_alpha(adj: np.ndarray) -> int:
    n = int(adj.shape[0])
    best = 0
    for mask in range(1 << n):
        vs = [i for i in range(n) if mask >> i & 1]
        if len(vs) <= best:
            continue
        ok = all(adj[i, j] == 0 for i, j in itertools.combinations(vs, 2))
        if ok:
            best = len(vs)
    return best


def bench_hoffman_bound(seed: int = 5311) -> dict[str, float]:
    # 4-regular circulant graph on 12 nodes: edges to +-1, +-2 neighbors
    n = 12
    adj = np.zeros((n, n))
    for i in range(n):
        for step in (1, 2):
            adj[i, (i + step) % n] = adj[i, (i - step) % n] = 1.0
    w = np.linalg.eigvalsh(adj)
    lam_min = float(w[0])
    deg = float(adj.sum(1)[0])
    bound = n * (-lam_min) / (deg - lam_min)
    alpha = _brute_alpha(adj)
    return {
        "synthetic_hb_bound": bound,
        "synthetic_hb_alpha": float(alpha),
        "synthetic_hb_lam_min": lam_min,
        "synthetic_hb_valid": float(alpha <= np.floor(bound + 1e-9) or alpha <= bound + 1e-9),
    }
