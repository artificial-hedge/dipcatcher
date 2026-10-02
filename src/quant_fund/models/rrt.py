"""RRT canon: rapidly-exploring random tree + RRT* rewiring in a
2-D configuration space with circular obstacles — steer toward
random samples with a fixed step, connect to goal when within
tolerance. Bench: feasibility rate over seeds, path cost, and
the RRT* improvement gap. All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _seg_free(a: FloatArray, b: FloatArray, obstacles: list[tuple[float, float, float]]) -> bool:
    """Segment-circle collision check by sampling."""
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


def rrt(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    bounds: tuple[float, float, float, float] = (0, 1, 0, 1),
    step: float = 0.05,
    max_iter: int = 4000,
    goal_tol: float = 0.08,
    seed: int = 0,
    star: bool = False,
    rewire_radius: float = 0.15,
) -> tuple[float, list[tuple[float, float]], int]:
    """RRT (star=False) or RRT* (star=True).

    Returns (path_length, waypoints, iters_used); length inf on
    failure.
    """
    rng = np.random.default_rng(seed)
    nodes = [np.asarray(start, dtype=np.float64)]
    parents = [-1]
    costs = [0.0]
    xmin, xmax, ymin, ymax = bounds
    goal_arr = np.asarray(goal, dtype=np.float64)
    goal_idx = -1
    for _it in range(max_iter):
        if rng.random() < 0.1:
            sample = goal_arr
        else:
            sample = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        d = [math.hypot(n[0] - sample[0], n[1] - sample[1]) for n in nodes]
        ni = int(np.argmin(d))
        near = nodes[ni]
        dist = d[ni]
        if dist < 1e-12:
            continue
        s = min(step, dist) / dist
        new = near + (sample - near) * s
        if not _seg_free(near, new, obstacles):
            continue
        parent = ni
        cost = costs[ni] + float(np.linalg.norm(new - near))
        if star:
            # choose cheapest parent within rewire radius
            cand = [
                j
                for j, nn in enumerate(nodes)
                if np.linalg.norm(nn - new) < rewire_radius and _seg_free(nn, new, obstacles)
            ]
            if cand:
                best = min(cand, key=lambda j: costs[j] + np.linalg.norm(nodes[j] - new))
                parent = best
                cost = costs[best] + float(np.linalg.norm(nodes[best] - new))
        nodes.append(new)
        parents.append(parent)
        costs.append(cost)
        idx = len(nodes) - 1
        if star:
            # rewire neighbors through the new node
            for j, nn in enumerate(nodes[:-1]):
                dd = float(np.linalg.norm(nn - new))
                if (
                    dd < rewire_radius
                    and costs[idx] + dd < costs[j]
                    and _seg_free(new, nn, obstacles)
                ):
                    parents[j] = idx
                    costs[j] = costs[idx] + dd
        if float(np.linalg.norm(new - goal_arr)) < goal_tol and _seg_free(new, goal_arr, obstacles):
            nodes.append(goal_arr)
            parents.append(idx)
            costs.append(costs[idx] + float(np.linalg.norm(new - goal_arr)))
            goal_idx = len(nodes) - 1
            break
    if goal_idx < 0:
        return math.inf, [], max_iter
    path = [(float(nodes[goal_idx][0]), float(nodes[goal_idx][1]))]
    c = goal_idx
    while parents[c] >= 0:
        c = parents[c]
        path.append((float(nodes[c][0]), float(nodes[c][1])))
    path.reverse()
    return costs[goal_idx], path, _it + 1


def bench_rrt(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    obstacles = [(0.5, 0.5, 0.12), (0.3, 0.75, 0.08), (0.7, 0.25, 0.08)]
    # feasibility across seeds
    wins = 0
    costs = []
    for s in range(8):
        L, path, it = rrt(
            (0.05, 0.05),
            (0.95, 0.95),
            obstacles,
            seed=seed + s,
            max_iter=3000,
        )
        if math.isfinite(L):
            wins += 1
            costs.append(L)
    out["synthetic_rrt_feasible_rate"] = wins / 8.0
    out["synthetic_rrt_mean_cost"] = float(np.mean(costs)) if costs else math.inf
    out["synthetic_rrt_min_cost"] = float(min(costs)) if costs else math.inf
    # RRT* cost improvement over the same budget
    Ls, _, _ = rrt(
        (0.05, 0.05),
        (0.95, 0.95),
        obstacles,
        seed=seed,
        max_iter=3000,
        star=True,
    )
    out["synthetic_rrtstar_cost"] = Ls
    out["synthetic_rrtstar_gap"] = out["synthetic_rrt_min_cost"] - Ls
    # obstacle-free sanity: straight-line ≈ optimal
    L_free, _, _ = rrt((0.1, 0.1), (0.9, 0.9), [], seed=seed, max_iter=2000)
    out["synthetic_rrt_free_ratio"] = L_free / math.hypot(0.8, 0.8)
    return out
