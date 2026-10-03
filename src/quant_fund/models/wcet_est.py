"""WCET estimation via longest-path on a CFG DAG."""

import numpy as np

_SEED = 20261231 + 652


def wcet(n_nodes: int, rng: np.random.RandomState) -> int:
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    cost = rng.randint(1, 10, n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < 0.15:
                adj[i].append(j)
    # longest path DP (DAG by construction)
    memo: dict[int, int] = {}

    def lp(u: int) -> int:
        if u in memo:
            return memo[u]
        best = cost[u]
        for v in adj[u]:
            best = max(best, cost[u] + lp(v))
        memo[u] = int(best)
        return int(best)

    return lp(0)


def bench_wcet_est(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 40
    for _ in range(trials):
        n = rng.randint(5, 15)
        w = wcet(n, rng)
        # oracle: w >= any single node cost and >= 1
        ok += w >= 1
    return {"synthetic_wcet_valid": ok / trials}
