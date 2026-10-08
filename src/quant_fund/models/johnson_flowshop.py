"""Johnson's rule (1954): optimal 2-machine flow-shop sequencing (SYNTHETIC).
Makespan vs random-order + SPT baselines; certified by exhaustive check
on small instances.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from quant_fund.models._sched_synth import flowshop, makespan


def _johnson(p: np.ndarray) -> list[int]:
    n = p.shape[0]
    left: list[int] = []
    right: list[int] = []
    remaining = set(range(n))
    while remaining:
        i = min(remaining, key=lambda j: min(p[j, 0], p[j, 1]))
        remaining.discard(i)
        if p[i, 0] <= p[i, 1]:
            left.append(i)
        else:
            right.insert(0, i)
    return left + right


def bench_johnson_flowshop(seed: int = 3041) -> dict[str, float]:
    p = flowshop(seed, n_jobs=9)
    order = _johnson(p)
    ms_j = makespan(order, p)
    rng = np.random.default_rng(seed)
    rnd = makespan(list(rng.permutation(9)), p)
    spt = makespan(sorted(range(9), key=lambda j: p[j].sum()), p)
    # exhaustive optimum on 9! = 362880 perms
    best = min(makespan(list(o), p) for o in permutations(range(9)))
    return {
        "synthetic_johnson_ms": ms_j,
        "synthetic_optimal_ms": best,
        "synthetic_random_ms": float(rnd),
        "synthetic_spt_ms": float(spt),
        "synthetic_johnson_gap": float(ms_j - best),
        "synthetic_johnson_vs_random": float(rnd - ms_j),
        "synthetic_torch_available": 0.0,
    }
