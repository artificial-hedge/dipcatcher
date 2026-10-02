"""Assignment canon: Hungarian algorithm + Hopcroft-Karp.

- ``hungarian`` — O(n^3) Kuhn-Munkres on a square cost matrix;
  returns the min-cost perfect matching (Jonker-Volgenant-style
  potential updates).
- ``hopcroft_karp`` — O(E sqrt(V)) maximum-cardinality matching
  on a bipartite graph via BFS layered augmenting paths.
- ``brute_assignment`` — exhaustive reference for small n.

Bench: planted assignment recovery vs brute force + parity
between Hungarian cost and brute minimum (SYNTHETIC only).
"""

from __future__ import annotations

from collections import deque
from itertools import permutations

import numpy as np

FloatArray = np.ndarray
IntArray = np.ndarray


def hungarian(cost: FloatArray) -> tuple[IntArray, float]:
    """Kuhn-Munkres for a square cost matrix (1-indexed potentials)."""
    a = np.asarray(cost, dtype=np.float64)
    n = a.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)  # p[j] = row matched to column j
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, j1 = p[j0], 0
            delta = np.inf
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    match = np.zeros(n, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] > 0:
            match[p[j] - 1] = j - 1
    return match, float(-v[0])


def hopcroft_karp(nl: int, nr: int, edges: list[tuple[int, int]]) -> tuple[int, IntArray]:
    g: list[list[int]] = [[] for _ in range(nl)]
    for u, v in edges:
        g[u].append(v)
    ml = np.full(nl, -1, dtype=np.int64)
    mr = np.full(nr, -1, dtype=np.int64)
    while True:
        dist = np.full(nl, -1, dtype=np.int64)
        dq: deque[int] = deque()
        for u in range(nl):
            if ml[u] < 0:
                dist[u] = 0
                dq.append(u)
        found = False
        while dq:
            u = dq.popleft()
            for v in g[u]:
                w = mr[v]
                if w < 0:
                    found = True
                elif dist[w] < 0:
                    dist[w] = dist[u] + 1
                    dq.append(w)

        def dfs(u: int, dist: np.ndarray = dist) -> bool:  # noqa: B023
            for v in g[u]:
                w = mr[v]
                if w < 0 or (dist[w] == dist[u] + 1 and dfs(w)):
                    ml[u], mr[v] = v, u
                    return True
            dist[u] = -1
            return False

        if not found:
            break
        for u in range(nl):
            if ml[u] < 0:
                dfs(u)
    return int((ml >= 0).sum()), ml


def brute_assignment(cost: FloatArray) -> float:
    n = cost.shape[0]
    return float(min(sum(cost[i, p[i]] for i in range(n)) for p in permutations(range(n))))


def bench_assignment(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 7
    cost = rng.random((n, n)) * 10
    match, val = hungarian(cost)
    brute = brute_assignment(cost)
    nl, nr = 30, 34
    edges = list({(int(u), int(v)) for u, v in rng.integers(0, (nl, nr), (120, 2))})
    sz, ml = hopcroft_karp(nl, nr, edges)
    # matching validity: no two lefts share a right
    used = set()
    valid = True
    for v in ml:
        if v >= 0:
            if v in used:
                valid = False
            used.add(v)
    return {
        "synthetic_hungarian_cost": val,
        "synthetic_brute_cost": brute,
        "synthetic_hungarian_err": abs(val - brute),
        "synthetic_hk_size": float(sz),
        "synthetic_hk_valid": float(valid),
        "synthetic_hk_matched": float((ml >= 0).sum()),
    }


__all__ = [
    "bench_assignment",
    "brute_assignment",
    "hopcroft_karp",
    "hungarian",
]
