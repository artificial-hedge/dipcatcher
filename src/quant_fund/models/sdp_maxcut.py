"""Goemans-Williamson max-cut via Burer-Monteiro SDP factorization.

X = YY^T with unit-norm rows replaces the SDP; projected gradient
ascent on sum_{(i,j) in E} (1 - <y_i, y_j>)/2, then random-hyperplane
rounding (200 draws, best kept). Bench reports the cut vs brute-force
optimum and vs the 0.878 guarantee threshold.
"""

import numpy as np

from quant_fund.models._approx_synth import G_EDGES, G_N, brute_maxcut


def _gw(seed: int = 7, iters: int = 300) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(G_N, 3))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    adj: list[list[int]] = [[] for _ in range(G_N)]
    for a, b in G_EDGES:
        adj[a].append(b)
        adj[b].append(a)
    lr = 0.1
    for _ in range(iters):
        grad = np.zeros_like(y)
        for v in range(G_N):
            grad[v] = -sum(y[u] for u in adj[v])
        y = y + lr * (grad - (grad * y).sum(1, keepdims=True) * y)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    best = 0
    for _ in range(200):
        g = rng.normal(size=3)
        side = (y @ g) > 0
        cut = sum(side[u] != side[v] for u, v in G_EDGES)
        best = max(best, cut)
    return best, y


def bench_sdp_maxcut(seed: int = 5301) -> dict[str, float]:
    cut, _ = _gw(seed)
    truth = brute_maxcut(G_EDGES, G_N)
    return {
        "synthetic_gw_cut": float(cut),
        "synthetic_gw_truth": float(truth),
        "synthetic_gw_ratio": cut / truth,
        "synthetic_gw_guarantee": 0.878,
    }
