"""SYNTHETIC Dinic max-flow with level graphs + blocking flow.

O(V²E) Dinic on integer-capacity digraphs; max-flow value verified against
LP-style brute min-cut enumeration on small graphs.
"""

from __future__ import annotations

import random
from collections import deque
from itertools import combinations


def dinic(n: int, edges: list[tuple[int, int, int]], s: int, t: int) -> int:
    cap: list[list[int]] = [[0] * n for _ in range(n)]
    for u, v, c in edges:
        cap[u][v] += c
    flow = 0
    while True:
        # BFS level graph
        level = [-1] * n
        level[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in range(n):
                if cap[u][v] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    dq.append(v)
        if level[t] < 0:
            return flow
        it = [0] * n
        while (f := _dfs(s, 10**9, cap, level, it, n, t)) > 0:
            flow += f


def _dfs(
    u: int,
    f: int,
    cap: list[list[int]],
    level: list[int],
    it: list[int],
    n: int,
    t: int,
) -> int:
    if u == t:
        return f
    for i in range(it[u], n):
        it[u] = i
        v = i
        if cap[u][v] > 0 and level[v] == level[u] + 1:
            d = _dfs(v, min(f, cap[u][v]), cap, level, it, n, t)
            if d > 0:
                cap[u][v] -= d
                cap[v][u] += d
                return d
    return 0


def _mincut_brute(n: int, edges: list[tuple[int, int, int]], s: int, t: int) -> int:
    """Exact s-t min-cut value by enumerating bipartitions (small n)."""
    nodes = [v for v in range(n) if v not in (s, t)]
    best = None
    for r in range(len(nodes) + 1):
        for sub in combinations(nodes, r):
            S = {s} | set(sub)
            T = set(range(n)) - S
            if t not in T:
                continue
            cut = sum(c for u, v, c in edges if u in S and v in T)
            best = cut if best is None else min(best, cut)
    return best if best is not None else 0


def bench_dinic_flow(seed: int = 20261231 + 520) -> dict[str, float]:
    rng = random.Random(seed)
    exact = 0
    n_trials = 40
    for _ in range(n_trials):
        n = rng.randrange(4, 8)
        s, t = 0, n - 1
        edges = []
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.35:
                    edges.append((u, v, rng.randrange(1, 10)))
        f = dinic(n, edges, s, t)
        exact += int(f == _mincut_brute(n, edges, s, t))
    # capacity conservation: flow ≤ total out-capacity of s
    bounded = True
    for _ in range(20):
        n = 6
        edges = [
            (u, v, rng.randrange(1, 8))
            for u in range(n)
            for v in range(n)
            if u != v and rng.random() < 0.4
        ]
        f = dinic(n, edges, 0, n - 1)
        bounded = bounded and f <= sum(c for u, v, c in edges if u == 0)
    return {
        "synthetic_maxflow_mincut": exact / n_trials,
        "synthetic_flow_bounded": float(bounded),
    }
