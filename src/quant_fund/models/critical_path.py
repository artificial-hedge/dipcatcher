"""SYNTHETIC CPM critical-path analysis on DAGs.

Forward/backward pass computes earliest/latest times + total float; longest
path verified against exhaustive path enumeration.
"""

from __future__ import annotations

import random
from collections import deque


def cpm(n: int, edges: list[tuple[int, int, float]]) -> tuple[float, dict[int, float], list[int]]:
    """Returns (project_length, total_float per node, critical path nodes)."""
    succ: dict[int, list[tuple[int, float]]] = {}
    pred: dict[int, list[int]] = {i: [] for i in range(n)}
    indeg = [0] * n
    for u, v, w in edges:
        succ.setdefault(u, []).append((v, w))
        pred[v].append(u)
        indeg[v] += 1
    # topo order
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in succ.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    es = [0.0] * n
    prev = [-1] * n
    for u in order:
        for v, w in succ.get(u, []):
            if es[u] + w > es[v]:
                es[v] = es[u] + w
                prev[v] = u
    t = max(range(n), key=lambda i: es[i])
    length = es[t]
    lf = [length] * n
    for u in reversed(order):
        for v, w in succ.get(u, []):
            lf[u] = min(lf[u], lf[v] - w)
    tf = {i: lf[i] - es[i] for i in range(n)}
    path = []
    u = t
    while u >= 0:
        path.append(u)
        u = prev[u]
    return length, tf, path[::-1]


def _longest_brute(n: int, edges: list[tuple[int, int, float]]) -> float:
    succ: dict[int, list[tuple[int, float]]] = {}
    for u, v, w in edges:
        succ.setdefault(u, []).append((v, w))
    best = 0.0

    def dfs(u: int, acc: float):
        nonlocal best
        best = max(best, acc)
        for v, w in succ.get(u, []):
            dfs(v, acc + w)

    starts = set(range(n)) - {v for u, v, _ in edges}
    for s0 in starts:
        dfs(s0, 0.0)
    return best


def bench_critical_path(seed: int = 20261231 + 524) -> dict[str, float]:
    rng = random.Random(seed)
    exact = 0
    slack_ok = 0
    n_trials = 40
    for _ in range(n_trials):
        n = rng.randrange(5, 11)
        edges = [
            (u, v, rng.uniform(0.5, 5))
            for u in range(n)
            for v in range(u + 1, n)
            if rng.random() < 0.35
        ]
        length, tf, path = cpm(n, edges)
        exact += int(abs(length - _longest_brute(n, edges)) < 1e-9)
        # critical-path nodes have zero total float and path weight = length
        w = {(u, v): w for u, v, w in edges}
        pw = sum(w[(a, b)] for a, b in zip(path, path[1:], strict=False))
        slack_ok += int(abs(pw - length) < 1e-9 and all(tf.get(u, 1) < 1e-9 for u in path))
    return {
        "synthetic_longest_exact": exact / n_trials,
        "synthetic_slack_consistent": slack_ok / n_trials,
    }
