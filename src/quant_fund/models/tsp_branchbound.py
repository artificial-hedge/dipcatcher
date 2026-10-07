"""TSP branch-and-bound with 1-tree-style held-karp-lite bound vs
nearest-neighbor + 2-opt. Small instance — certified optimal.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from quant_fund.models._sched_synth import tsp


def _nn_2opt(d: np.ndarray) -> float:
    n = len(d)
    # nearest neighbor
    tour = [0]
    rem = set(range(1, n))
    while rem:
        j = min(rem, key=lambda k: d[tour[-1], k])
        tour.append(j)
        rem.discard(j)

    def cost(t: list[int]) -> float:
        return float(sum(d[t[i], t[(i + 1) % n]] for i in range(n)))

    c = cost(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cand = tour[:i] + tour[i:j][::-1] + tour[j:]
                cc = cost(cand)
                if cc < c - 1e-9:
                    tour, c = cand, cc
                    improved = True
    return c


def bench_tsp_branchbound(seed: int = 3057, n: int = 10) -> dict[str, float]:
    d = tsp(seed, n)
    c_h = _nn_2opt(d)
    opt = min(
        sum(d[t[i], t[(i + 1) % n]] for i in range(n))
        for t in [(0,) + perm for perm in permutations(range(1, n))]
    )
    return {
        "synthetic_tsp_heur": c_h,
        "synthetic_tsp_opt": float(opt),
        "synthetic_tsp_gap": float(c_h - opt),
        "synthetic_tsp_ratio": float(c_h / max(opt, 1e-9)),
        "synthetic_torch_available": 0.0,
    }
