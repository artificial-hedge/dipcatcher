"""NEH heuristic (Nawaz et al. 1983) for m-machine permutation flow-shop
— insertion-improved ordering vs Johnson/random baselines.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sched_synth import flowshop, makespan


def _neh(p: np.ndarray) -> list[int]:
    tot = p.sum(1)
    jobs = sorted(range(len(tot)), key=lambda j: -tot[j])
    seq: list[int] = []
    for j in jobs:
        best_i, best_ms = 0, np.inf
        for i in range(len(seq) + 1):
            cand = seq[:i] + [j] + seq[i:]
            ms = makespan(cand, p)
            if ms < best_ms:
                best_ms, best_i = ms, i
        seq = seq[:best_i] + [j] + seq[best_i:]
    return seq


def bench_neh_heuristic(seed: int = 3045) -> dict[str, float]:
    p = flowshop(seed, n_jobs=10, n_mach=4)
    ms_n = makespan(_neh(p), p)
    rng = np.random.default_rng(seed)
    rnd = np.mean([makespan(list(rng.permutation(10)), p) for _ in range(50)])
    return {
        "synthetic_neh_ms": ms_n,
        "synthetic_neh_random_mean": float(rnd),
        "synthetic_neh_gain": float(rnd - ms_n),
        "torch_available": 0.0,
    }
