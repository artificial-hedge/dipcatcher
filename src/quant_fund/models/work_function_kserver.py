"""Work-function algorithm for k-server on a line metric (SYNTHETIC).

w_t(S) = min over configs reachable... standard DP over request
sequence on a 5-point line with k=2 servers. Bench compares the online
cost vs the brute-force offline optimum on a fixed request string.
"""

import itertools

import numpy as np


def _points() -> np.ndarray:
    return np.array([0.0, 2.0, 5.0, 7.0, 10.0])


def _requests() -> list[int]:
    return [1, 3, 0, 4, 2, 1, 4, 0, 3, 2]


def _wfa(k: int = 2) -> tuple[float, float]:
    pts = _points()
    reqs = _requests()
    configs = list(itertools.combinations(range(len(pts)), k))
    dist = np.abs(pts[:, None] - pts[None, :])
    n_c = len(configs)
    # W_t[C] = min over C' of W_{t-1}[C'] + min-cost transport C'->C
    w = np.zeros(n_c)
    tot_online = 0.0
    cur: tuple[int, ...] = (0, 1)
    for r in reqs:
        new_w = np.full(n_c, np.inf)
        for cprime_i, cprime in enumerate(configs):
            for c_i, c in enumerate(configs):
                # move one server in c' to cover r resulting in c
                if r in c:
                    # transport cost: min over matchings c'->c
                    move = _min_transport(cprime, c, dist)
                    cand = w[cprime_i] + move
                    new_w[c_i] = min(new_w[c_i], cand)
        w = new_w
        # online move: choose config minimizing w[c] + actual move from cur
        best_val = np.inf
        best_c: tuple[int, ...] = cur
        for c_i, c in enumerate(configs):
            if r in c:
                val = float(w[c_i] + _min_transport(cur, c, dist))
                if val < best_val:
                    best_val, best_c = val, c
        tot_online += _min_transport(cur, best_c, dist)
        cur = best_c
    offline = float(w.min())
    return tot_online, offline


def _min_transport(a: tuple[int, ...], b: tuple[int, ...], dist: np.ndarray) -> float:
    # both size k: min over bijections of sum dist
    best = np.inf
    for perm in itertools.permutations(b):
        best = min(best, sum(dist[x, y] for x, y in zip(a, perm, strict=True)))
    return float(best)


def bench_work_function_kserver(seed: int = 5405) -> dict[str, float]:
    online, offline = _wfa()
    return {
        "synthetic_wfa_online": online,
        "synthetic_wfa_offline": offline,
        "synthetic_wfa_ratio": online / offline if offline > 0 else 1.0,
        "synthetic_wfa_bound": float(2 * 2 - 1),
    }
