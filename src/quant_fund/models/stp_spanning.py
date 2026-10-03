"""Spanning-tree protocol: root election + loop-free path selection."""

import numpy as np

_SEED = 20261231 + 711


def stp_run(adj: dict[int, set[int]], n: int) -> tuple[int, int]:
    """Return (root_id, n_tree_edges). Root = min id; tree via BFS."""
    root = min(adj.keys())
    parent = {root: -1}
    q = [root]
    while q:
        u = q.pop(0)
        for v in sorted(adj.get(u, ())):
            if v not in parent:
                parent[v] = u
                q.append(v)
    return root, len(parent) - 1


def bench_stp_spanning(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = int(rng.randint(4, 9))
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        # ensure connected: chain + extra edges
        for i in range(n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
        for i in range(n):
            for j in range(i + 2, n):
                if rng.rand() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        root, edges = stp_run(adj, n)
        ok += float(root == 0 and edges == n - 1)
    return {"synthetic_stp_tree": ok / trials}
