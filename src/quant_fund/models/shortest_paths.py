"""Shortest-path canon: Dijkstra, Bellman-Ford, A*, Floyd-Warshall.

- ``dijkstra`` — binary-heap O((V+E)log V) for nonneg weights.
- ``bellman_ford`` — O(VE) with negative-edge support and
  negative-cycle detection.
- ``a_star`` — consistent-heuristic A* (optimal).
- ``floyd_warshall`` — all-pairs O(V^3) dense matrix form.

Bench: planted shortest path vs dense Floyd-Warshall reference
(SYNTHETIC only).
"""

from __future__ import annotations

import heapq

import numpy as np

FloatArray = np.ndarray


def dijkstra(n: int, edges: list[tuple[int, int, float]], src: int = 0) -> FloatArray:
    g: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        g[u].append((v, w))
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def bellman_ford(
    n: int, edges: list[tuple[int, int, float]], src: int = 0
) -> tuple[FloatArray, bool]:
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            if np.isfinite(dist[u]) and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    neg_cycle = any(np.isfinite(dist[u]) and dist[u] + w < dist[v] for u, v, w in edges)
    return dist, neg_cycle


def a_star(
    n: int,
    edges: list[tuple[int, int, float]],
    src: int,
    dst: int,
    heuristic,
) -> tuple[float, list[int]]:
    g: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        g[u].append((v, w))
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    prev = np.full(n, -1)
    pq = [(0.0, src)]
    while pq:
        f, u = heapq.heappop(pq)
        if u == dst:
            path = [dst]
            while prev[path[-1]] >= 0:
                path.append(int(prev[path[-1]]))
            return float(dist[dst]), path[::-1]
        if f - float(heuristic(u)) > dist[u]:
            continue
        for v, w in g[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + float(heuristic(v)), v))
    return float("inf"), []


def floyd_warshall(n: int, edges: list[tuple[int, int, float]]) -> FloatArray:
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    for u, v, w in edges:
        d[u, v] = min(d[u, v], w)
    for k in range(n):
        d = np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    return d


def bench_shortest_paths(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 80
    edges: list[tuple[int, int, float]] = []
    for _ in range(320):
        u, v = rng.integers(0, n, 2)
        if u != v:
            edges.append((int(u), int(v), float(rng.random() * 5 + 0.1)))
    dij = dijkstra(n, edges, 0)
    bf, neg = bellman_ford(n, edges, 0)
    fw = floyd_warshall(n, edges)
    fin = np.isfinite(fw[0])
    err_dij = float(np.max(np.abs(dij[fin] - fw[0][fin])))
    err_bf = float(np.max(np.abs(bf[fin] - fw[0][fin])))
    # A* with zero heuristic = Dijkstra (consistent)
    d_a, path = a_star(n, edges, 0, n - 1, lambda u: 0.0)
    err_a = abs(d_a - float(fw[0, n - 1]))
    return {
        "synthetic_dijkstra_err": err_dij,
        "synthetic_bellman_err": err_bf,
        "synthetic_neg_cycle": float(neg),
        "synthetic_astar_err": float(err_a),
        "synthetic_astar_path_len": float(len(path)),
    }


__all__ = [
    "a_star",
    "bellman_ford",
    "bench_shortest_paths",
    "dijkstra",
    "floyd_warshall",
]
