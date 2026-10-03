"""PRM canon: probabilistic roadmap — uniform milestone sampling,
k-nearest local-planner edges, A*/Dijkstra query over the
resulting graph against circular obstacles. Compared with the
continuous straight-line lower bound and its own grid-free
baseline. All SYNTHETIC.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _seg_free(a: FloatArray, b: FloatArray, obstacles: list[tuple[float, float, float]]) -> bool:
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(d / 0.02))
    for i in range(n + 1):
        t = i / n
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        for ox, oy, r in obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r * r:
                return False
    return True


def prm_build(
    n_samples: int,
    obstacles: list[tuple[float, float, float]],
    k: int = 8,
    bounds: tuple[float, float, float, float] = (0, 1, 0, 1),
    seed: int = 0,
) -> tuple[FloatArray, list[list[tuple[int, float]]]]:
    """Sample milestones, connect k nearest within collision
    radius — returns (milestones, adjacency [(j, cost)])."""
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    pts: list[tuple[float, float]] = []
    for _ in range(n_samples * 4):  # oversample for rejections
        if len(pts) >= n_samples:
            break
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if any((x - ox) ** 2 + (y - oy) ** 2 <= r * r for ox, oy, r in obstacles):
            continue
        pts.append((x, y))
    P = np.asarray(pts, dtype=np.float64)
    n = len(P)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        d = np.linalg.norm(P - P[i], axis=1)
        nbrs = np.argsort(d)[1 : k + 1]
        for j in nbrs:
            c = float(d[j])
            if _seg_free(P[i], P[j], obstacles):
                adj[i].append((int(j), c))
                adj[int(j)].append((i, c))
    return P, adj


def prm_query(
    P: FloatArray,
    adj: list[list[tuple[int, float]]],
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    connect: int = 6,
) -> tuple[float, int]:
    """Dijkstra query — connect start/goal into the roadmap then
    shortest path. Returns (cost, nodes_expanded)."""
    s = np.asarray(start)
    g = np.asarray(goal)
    ds = np.linalg.norm(P - s, axis=1)
    dg = np.linalg.norm(P - g, axis=1)
    s_nodes = [(i, float(ds[i])) for i in np.argsort(ds)[:connect] if _seg_free(s, P[i], obstacles)]
    g_nodes = [(i, float(dg[i])) for i in np.argsort(dg)[:connect] if _seg_free(P[i], g, obstacles)]
    dist = {i: c for i, c in s_nodes}
    heap = [(c, i) for i, c in s_nodes]
    heapq.heapify(heap)
    best_goal = math.inf
    expanded = 0
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist.get(i, math.inf):
            continue
        expanded += 1
        gcost = next((c for j, c in g_nodes if j == i), None)
        if gcost is not None and d + gcost < best_goal:
            best_goal = d + gcost
        for j, w in adj[i]:
            nd = d + w
            if nd < dist.get(j, math.inf):
                dist[j] = nd
                heapq.heappush(heap, (nd, j))
    return best_goal, expanded


def bench_prm(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    obstacles = [(0.5, 0.5, 0.15), (0.25, 0.7, 0.08)]
    P, adj = prm_build(400, obstacles, k=8, seed=seed)
    out["synthetic_prm_milestones"] = float(len(P))
    out["synthetic_prm_edge_density"] = float(sum(len(a) for a in adj) / (2 * len(P)))
    cost, expanded = prm_query(P, adj, (0.05, 0.05), (0.95, 0.95), obstacles)
    out["synthetic_prm_cost"] = cost
    out["synthetic_prm_expanded"] = float(expanded)
    # lower bound = straight-line distance (unreachable but bounded)
    lb = math.hypot(0.9, 0.9)
    out["synthetic_prm_lb_ratio"] = cost / lb
    # denser roadmap → better cost (second roadmap, 2x samples)
    P2, adj2 = prm_build(800, obstacles, k=10, seed=seed + 7)
    cost2, _ = prm_query(P2, adj2, (0.05, 0.05), (0.95, 0.95), obstacles)
    out["synthetic_prm_dense_cost"] = cost2
    out["synthetic_prm_density_gain"] = cost - cost2
    out["synthetic_prm_feasible"] = float(math.isfinite(cost) and math.isfinite(cost2))
    return out
