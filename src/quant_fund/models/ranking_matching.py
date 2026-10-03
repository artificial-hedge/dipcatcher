"""RANKING online bipartite matching (1 - 1/e competitive).

Offline vertices get a uniform random priority; online vertex matches
to its highest-priority free neighbor. Bench averages over priority
draws on a small bipartite fixture vs the brute-force max matching.
"""

import numpy as np

_L = [0, 1, 2, 3]
_R = [0, 1, 2, 3]
_EDGES = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)]


def _max_matching() -> int:
    best = 0
    for mask in range(1 << len(_EDGES)):
        ls = [_EDGES[i][0] for i in range(len(_EDGES)) if mask >> i & 1]
        rs = [_EDGES[i][1] for i in range(len(_EDGES)) if mask >> i & 1]
        if len(set(ls)) == len(ls) and len(set(rs)) == len(rs):
            best = max(best, len(ls))
    return best


def _ranking(seed: int, trials: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    order_r = list(range(len(_R)))
    tot = 0
    adj: dict[int, list[int]] = {left: [] for left in _L}
    for left, r in _EDGES:
        adj[left].append(r)
    for _ in range(trials):
        prio = rng.permutation(order_r)
        rank = {r: i for i, r in enumerate(prio)}
        free = set(order_r)
        m = 0
        for left in rng.permutation(_L):
            cand = [r for r in adj[left] if r in free]
            if cand:
                pick = min(cand, key=lambda r: rank[r])
                free.discard(pick)
                m += 1
        tot += m
    return tot / trials


def bench_ranking_matching(seed: int = 5407) -> dict[str, float]:
    avg = _ranking(seed)
    truth = _max_matching()
    return {
        "synthetic_rk_avg": avg,
        "synthetic_rk_truth": float(truth),
        "synthetic_rk_ratio": avg / truth,
        "synthetic_rk_bound": 1.0 - 1.0 / np.e,
    }
