"""WCET estimation via longest-path on a CFG DAG (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 652


def _longest_path(adj: dict[int, list[int]], cost: np.ndarray, start: int = 0) -> int:
    """Longest path (sum of node costs) from ``start`` on a DAG."""
    memo: dict[int, int] = {}

    def lp(u: int) -> int:
        if u in memo:
            return memo[u]
        best = int(cost[u])
        for v in adj.get(u, []):
            best = max(best, int(cost[u]) + lp(v))
        memo[u] = int(best)
        return int(best)

    return lp(start)


def wcet(n_nodes: int, rng: np.random.RandomState) -> int:
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    cost = rng.randint(1, 10, n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < 0.15:
                adj[i].append(j)
    return _longest_path(adj, cost)


def bench_wcet_est(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 40
    for _ in range(trials):
        n = rng.randint(5, 15)
        w = wcet(n, rng)
        ok += w >= 1
    # oracle on crafted DAGs where the longest path is known: the
    # previous w >= 1 bound held by construction and could never fail.
    chain = {0: [1], 1: [2], 2: [3], 3: []}
    ok += int(_longest_path(chain, np.array([2, 5, 1, 7])) == 15)
    fork = {0: [1, 2], 1: [3], 2: [3], 3: []}
    ok += int(_longest_path(fork, np.array([1, 9, 3, 4])) == 14)
    leaf_only = {0: [], 1: [2], 2: []}
    ok += int(_longest_path(leaf_only, np.array([4, 9, 9])) == 4)
    return {"synthetic_wcet_valid": ok / (trials + 3)}
