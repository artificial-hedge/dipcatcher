"""SYNTHETIC A* search on grid + graph.

Manhattan-admissible A* on weighted grids verified optimal against Dijkstra;
expansion count bounded by Dijkstra's (consistent heuristic property).
"""

from __future__ import annotations

import heapq
import random


def astar(
    n: int,
    edges: dict[int, list[tuple[int, float]]],
    h: dict[int, float],
    s: int,
    t: int,
) -> tuple[float, int]:
    """A* with consistent heuristic; returns (cost, expansions)."""
    g = {s: 0.0}
    pq = [(h.get(s, 0.0), s)]
    done = set()
    expansions = 0
    while pq:
        f, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        expansions += 1
        if u == t:
            return g[u], expansions
        for v, w in edges.get(u, []):
            if v not in done:
                ng = g[u] + w
                if ng < g.get(v, 10**18):
                    g[v] = ng
                    heapq.heappush(pq, (ng + h.get(v, 0.0), v))
    return 10**18, expansions


def dijkstra(
    n: int, edges: dict[int, list[tuple[int, float]]], s: int, t: int
) -> tuple[float, int]:
    g = {s: 0.0}
    pq = [(0.0, s)]
    done = set()
    exp = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        exp += 1
        if u == t:
            return g[u], exp
        for v, w in edges.get(u, []):
            if v not in done:
                ng = g[u] + w
                if ng < g.get(v, 10**18):
                    g[v] = ng
                    heapq.heappush(pq, (ng, v))
    return 10**18, exp


def _grid(
    rng: random.Random, w: int, h: int, blocked: float
) -> tuple[dict[int, list[tuple[int, float]]], set[tuple[int, int]]]:
    edges: dict[int, list[tuple[int, float]]] = {}
    open_cells = {(x, y) for x in range(w) for y in range(h) if rng.random() > blocked}
    open_cells.add((0, 0))
    open_cells.add((w - 1, h - 1))
    for x, y in open_cells:
        u = y * w + x
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in open_cells:
                edges.setdefault(u, []).append((ny * w + nx, 1.0))
    return edges, open_cells


def _manhattan(w: int, t: int) -> dict[int, float]:
    tx, ty = t % w, t // w
    return {y * w + x: abs(x - tx) + abs(y - ty) for y in range(200) for x in range(200)}


def bench_astar_search(seed: int = 20261231 + 522) -> dict[str, float]:
    rng = random.Random(seed)
    optimal = 0
    fewer = 0
    n = 40
    for _ in range(n):
        w, h = 12, 12
        edges, _ = _grid(rng, w, h, 0.25)
        s, t = 0, w * h - 1
        hv = _manhattan(w, t)
        da, ea = astar(w * h, edges, hv, s, t)
        dd, ed = dijkstra(w * h, edges, s, t)
        optimal += int(da == dd)
        if dd < 10**17:
            fewer += int(ea <= ed)
        else:
            fewer += 1
    return {
        "synthetic_optimal_vs_dijkstra": optimal / n,
        "synthetic_expansions_bounded": fewer / n,
    }
