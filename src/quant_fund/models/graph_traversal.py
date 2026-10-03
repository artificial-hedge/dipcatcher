"""Graph traversal canon: BFS, DFS, topological sort, bipartite.

Adjacency-list oriented primitives used downstream by flow,
matching, and decomposition modules:

- ``bfs`` — breadth-first order + parent tree + levels.
- ``dfs_iter`` — iterative depth-first order.
- ``kahn_topo`` — Kahn's algorithm; raises on cycles.
- ``is_bipartite`` — 2-coloring check, returns the coloring.

Bench: layered DAG + odd-cycle graph checks (SYNTHETIC).
"""

from __future__ import annotations

from collections import deque

import numpy as np

IntArray = np.ndarray


def _adj(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
    return g


def bfs(n: int, edges: list[tuple[int, int]], src: int = 0) -> dict[str, IntArray]:
    g = _adj(n, edges)
    seen = np.full(n, -1)
    parent = np.full(n, -1)
    order: list[int] = []
    seen[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in g[u]:
            if seen[v] < 0:
                seen[v] = seen[u] + 1
                parent[v] = u
                dq.append(v)
    return {
        "order": np.array(order, dtype=np.int64),
        "level": seen,
        "parent": parent,
    }


def dfs_iter(n: int, edges: list[tuple[int, int]], src: int = 0) -> IntArray:
    g = _adj(n, edges)
    seen = np.zeros(n, dtype=bool)
    order: list[int] = []
    stack = [src]
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        stack.extend(reversed(g[u]))
    return np.array(order, dtype=np.int64)


def kahn_topo(n: int, edges: list[tuple[int, int]]) -> IntArray:
    g = _adj(n, edges)
    indeg = np.zeros(n, dtype=np.int64)
    for _, v in edges:
        indeg[v] += 1
    dq = deque(int(i) for i in np.flatnonzero(indeg == 0))
    order: list[int] = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    if len(order) != n:
        raise ValueError("graph has a cycle")
    return np.array(order, dtype=np.int64)


def is_bipartite(n: int, edges: list[tuple[int, int]]) -> tuple[bool, IntArray]:
    g = _adj(n, edges)
    for u, v in edges:
        g[v].append(u)  # undirected
    color = np.full(n, -1)
    for s in range(n):
        if color[s] >= 0:
            continue
        color[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in g[u]:
                if color[v] < 0:
                    color[v] = 1 - color[u]
                    dq.append(v)
                elif color[v] == color[u]:
                    return False, color
    return True, color


def bench_graph_traversal(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, layers = 60, 6
    nodes_per = n // layers
    edges: list[tuple[int, int]] = []
    for lv in range(layers - 1):
        for _ in range(14):
            u = lv * nodes_per + int(rng.integers(nodes_per))
            v = (lv + 1) * nodes_per + int(rng.integers(nodes_per))
            edges.append((u, v))
    topo = kahn_topo(n, edges)
    pos = np.empty(n, dtype=np.int64)
    pos[topo] = np.arange(n)
    ok = float(all(pos[u] < pos[v] for u, v in edges))
    r_bfs = bfs(n, edges, 0)
    dag_edges = [(u, v) for u, v in edges]
    # make an odd cycle for the bipartite negative case
    odd = [(0, 1), (1, 2), (2, 0)]
    ok_bip_pos, _ = is_bipartite(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    ok_bip_neg, _ = is_bipartite(5, odd)
    return {
        "synthetic_topo_valid": ok,
        "synthetic_bfs_depth": float(r_bfs["level"].max()),
        "synthetic_bip_ok": float(ok_bip_pos),
        "synthetic_bip_neg": float(not ok_bip_neg),
        "synthetic_dfs_reach": float(len(dfs_iter(n, dag_edges, 0))),
    }


__all__ = [
    "bench_graph_traversal",
    "bfs",
    "dfs_iter",
    "is_bipartite",
    "kahn_topo",
]
