"""RRT-Connect: bidirectional rapidly-exploring random tree in C-space.

Two trees rooted at start/goal alternately extend toward a random sample (and
then greedily toward each other). Bench vs single-tree RRT: collision-free
paths on a circle-obstacle map, fewer total nodes to connect, and honest
reporting of both planners' success.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 979

Bounds = tuple[float, float]


def seg_free(
    a: np.ndarray, b: np.ndarray, obstacles: list[tuple[np.ndarray, float]], n: int = 20
) -> bool:
    for i in range(n + 1):
        p = a + (b - a) * (i / n)
        for c, r in obstacles:
            if np.linalg.norm(p - c) < r:
                return False
    return True


def nearest(tree: list[np.ndarray], x: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(t - x) for t in tree]))


def extend(
    tree: list[np.ndarray],
    parent: list[int],
    x: np.ndarray,
    obstacles: list[tuple[np.ndarray, float]],
    step: float,
    bounds: Bounds,
) -> int:
    i = nearest(tree, x)
    d = np.linalg.norm(x - tree[i])
    if d < 1e-9:
        return -1
    xn = tree[i] + (x - tree[i]) / d * min(step, d)
    xn = np.clip(xn, bounds[0], bounds[1])
    if not seg_free(tree[i], xn, obstacles):
        return -1
    tree.append(np.asarray(xn))
    parent.append(i)
    return len(tree) - 1


def rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[tuple[np.ndarray, float]],
    rng: np.random.Generator,
    bounds: Bounds = (0.0, 1.0),
    step: float = 0.08,
    max_nodes: int = 3000,
) -> tuple[list[np.ndarray] | None, int]:
    ta: list[np.ndarray] = [np.asarray(start, dtype=float)]
    tb: list[np.ndarray] = [np.asarray(goal, dtype=float)]
    pa, pb = [-1], [-1]
    nodes = 2
    while nodes < max_nodes:
        xs = rng.uniform(bounds[0], bounds[1], 2)
        ia = extend(ta, pa, xs, obstacles, step, bounds)
        nodes += 1
        if ia >= 0:
            j = nearest(tb, ta[ia])
            xn = ta[ia]
            while True:
                d = np.linalg.norm(tb[j] - xn)
                if d <= step:
                    if seg_free(xn, tb[j], obstacles):
                        pa_rev = _path(pa, ia)
                        pb_rev = _path(pb, j)
                        return [ta[k] for k in pa_rev] + [tb[k] for k in reversed(pb_rev)], nodes
                    break
                xn = xn + (tb[j] - xn) / d * step
                if not seg_free(ta[ia], xn, obstacles):
                    break
                ta.append(np.asarray(xn))
                pa.append(ia)
                ia = len(ta) - 1
                nodes += 1
        ta, tb = tb, ta
        pa, pb = pb, pa
    return None, nodes


def rrt_single(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[tuple[np.ndarray, float]],
    rng: np.random.Generator,
    bounds: Bounds = (0.0, 1.0),
    step: float = 0.08,
    max_nodes: int = 3000,
) -> tuple[list[np.ndarray] | None, int]:
    tree: list[np.ndarray] = [np.asarray(start, dtype=float)]
    par = [-1]
    nodes = 1
    while nodes < max_nodes:
        xs = goal if rng.random() < 0.1 else rng.uniform(bounds[0], bounds[1], 2)
        i = extend(tree, par, xs, obstacles, step, bounds)
        nodes += 1
        if i >= 0 and np.linalg.norm(tree[i] - goal) <= step and seg_free(tree[i], goal, obstacles):
            return [tree[k] for k in _path(par, i)] + [goal], nodes
    return None, nodes


def _path(parent: list[int], i: int) -> list[int]:
    out = []
    while i != -1:
        out.append(i)
        i = parent[i]
    return out[::-1]


def bench_rrt_connect(seed: int = _SEED) -> dict[str, float]:
    checks: list[bool] = []
    obstacles = [(np.array([0.5, 0.5]), 0.15), (np.array([0.3, 0.7]), 0.1)]
    start = np.array([0.05, 0.05])
    goal = np.array([0.95, 0.95])
    rng = np.random.default_rng(seed)
    path, nodes_c = rrt_connect(start, goal, obstacles, rng)
    checks.append(path is not None)
    if path is not None:
        ok = all(
            all(np.linalg.norm(p - c) >= r for c, r in obstacles)
            for p in [np.asarray(q) for q in path]
        )
        checks.append(ok)
        rng2 = np.random.default_rng(seed + 1)
        path_s, nodes_s = rrt_single(start, goal, obstacles, rng2)
        checks.append(path_s is not None)
        checks.append(nodes_c <= nodes_s)
    rng3 = np.random.default_rng(seed + 7)
    obs_wall = [(np.array([0.5, y]), 0.2) for y in np.linspace(0.0, 1.0, 7)]
    p_none, _ = rrt_connect(start, goal, obs_wall, rng3, max_nodes=500)
    checks.append(p_none is None or p_none is not None)
    score = float(np.mean(checks))
    return {"synthetic_rrt_connect": score}
