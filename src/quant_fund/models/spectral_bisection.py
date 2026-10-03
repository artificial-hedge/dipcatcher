"""Fiedler-vector spectral bisection.

Partitions the graph by the sign of the second-smallest Laplacian
eigenvector; balanced by median split. Bench compares the bisection
cut against the brute-force min bisection on the shared graph.
"""

import itertools

import numpy as np

from quant_fund.models._approx_synth import G_EDGES, G_N


def _brute_bisect() -> int:
    best = len(G_EDGES) + 1
    half = G_N // 2
    for comb in itertools.combinations(range(1, G_N), half - 1):
        side = {0, *comb}
        c = sum((u in side) != (v in side) for u, v in G_EDGES)
        best = min(best, c)
    return best


def bench_spectral_bisection(seed: int = 5309) -> dict[str, float]:
    a = np.zeros((G_N, G_N))
    for u, v in G_EDGES:
        a[u, v] = a[v, u] = 1.0
    lap = np.diag(a.sum(1)) - a
    w, vec = np.linalg.eigh(lap)
    f = vec[:, 1]
    med = np.median(f)
    side = f > med
    # enforce exact half by thresholding at order stats
    order = np.argsort(f)
    side = np.zeros(G_N, dtype=bool)
    side[order[: G_N // 2]] = True
    cut = int(sum(side[u] != side[v] for u, v in G_EDGES))
    truth = _brute_bisect()
    return {
        "synthetic_sb_cut": float(cut),
        "synthetic_sb_truth": float(truth),
        "synthetic_sb_ratio": cut / truth,
        "synthetic_sb_fiedler": float(w[1]),
    }
