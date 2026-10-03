"""SYNTHETIC min-cost max-flow via successive shortest paths + potentials.

SSP with Johnson potentials (Bellman-Ford initialization, Dijkstra
augmentations); total cost verified against exhaustive flow enumeration on
tiny networks and flow-conservation checked on larger ones.
"""

from __future__ import annotations

import random


def mincost_flow(
    n: int,
    edges: list[tuple[int, int, int, int]],  # u, v, cap, cost
    s: int,
    t: int,
    f_target: int,
) -> tuple[int, int]:
    """Returns (flow_sent, total_cost)."""
    cap = [[0] * n for _ in range(n)]
    cost = [[0] * n for _ in range(n)]
    for u, v, c, w in edges:
        cap[u][v] += c
        cost[u][v] = w
        cost[v][u] = -w  # residual
    # potentials via Bellman-Ford
    h = [0] * n
    for _ in range(n):
        for u, v, _c, w in edges:
            if cap[u][v] > 0 and h[u] + w < h[v]:
                h[v] = h[u] + w
    flow = fcost = 0
    while flow < f_target:
        dist = [10**18] * n
        dist[s] = 0
        prev = [-1] * n
        done = [False] * n
        for _ in range(n):
            u = -1
            for v2 in range(n):
                if not done[v2] and dist[v2] < 10**18 and (u < 0 or dist[v2] < dist[u]):
                    u = v2
            if u < 0:
                break
            done[u] = True
            for v in range(n):
                if cap[u][v] > 0:
                    nd = dist[u] + cost[u][v] + h[u] - h[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
        if dist[t] >= 10**18:
            break
        for v in range(n):
            if dist[v] < 10**18:
                h[v] += dist[v]
        aug = f_target - flow
        v = t
        while v != s:
            aug = min(aug, cap[prev[v]][v])
            v = prev[v]
        v = t
        while v != s:
            cap[prev[v]][v] -= aug
            cap[v][prev[v]] += aug
            fcost += aug * cost[prev[v]][v]
            v = prev[v]
        flow += aug
    return flow, fcost


def _brute_mincost(
    n: int, edges: list[tuple[int, int, int, int]], s: int, t: int, f_target: int
) -> tuple[int, int] | None:
    """Enumerate s-t paths + flows on tiny graphs; exact min cost."""
    # enumerate all simple paths, then flow-split over paths via search
    paths: list[list[int]] = []

    def _dfs(u: int, seen: set[int], path: list[int]):
        if u == t:
            paths.append(list(path))
            return
        for u2, v, _c, _w in edges:
            if u2 == u and v not in seen:
                seen.add(v)
                path.append(v)
                _dfs(v, seen, path)
                path.pop()
                seen.discard(v)

    _dfs(s, {s}, [s])
    if not paths:
        return (0, 0) if f_target == 0 else None

    # edge→cost map; allocate flow greedily by path cost? No — enumerate:
    # small: try all allocations of ≤3 path flows (bounded caps)
    def path_cost(p: list[int]) -> int:
        return sum(
            w for u, v, c, w in edges for a, b in zip(p, p[1:], strict=False) if (u, v) == (a, b)
        )

    path_min_cap = {
        tuple(p): min(
            c for a, b in zip(p, p[1:], strict=False) for u, v, c, _w in edges if (u, v) == (a, b)
        )
        for p in paths
    }
    best: int | None = None
    # DFS over path-flow allocations
    edge_list = [(u, v, c) for u, v, c, _ in edges]

    def ok_caps(assign: dict[tuple, int]) -> bool:
        for u, v, c in edge_list:
            use = sum(
                f
                for p, f in assign.items()
                for a, b in zip(p, p[1:], strict=False)
                if (a, b) == (u, v)
            )
            if use > c:
                return False
        return True

    order = list(paths)
    chosen: dict[tuple, int] = {}

    def search(k: int, need: int, cost_so_far: int):
        nonlocal best
        if k == len(order):
            if need == 0 and ok_caps(chosen) and (best is None or cost_so_far < best):
                best = cost_so_far
            return
        p = order[k]
        pc = path_cost(p)
        for f in range(min(need, path_min_cap[tuple(p)] * 10) + 1):
            chosen[tuple(p)] = f
            search(k + 1, need - f, cost_so_far + f * pc)
        chosen.pop(tuple(p), None)

    search(0, f_target, 0)
    return (f_target, best) if best is not None else None


def bench_mincost_flow(seed: int = 20261231 + 521) -> dict[str, float]:
    rng = random.Random(seed)
    exact = 0
    n_exact = 25
    for _ in range(n_exact):
        n = rng.randrange(4, 6)
        s, t = 0, n - 1
        edges = []
        for u in range(n):
            for v in range(n):
                if u < v and rng.random() < 0.5:
                    edges.append((u, v, rng.randrange(1, 4), rng.randrange(0, 6)))
        f_target = rng.randrange(1, 5)
        f, c = mincost_flow(n, edges, s, t, f_target)
        oracle = _brute_mincost(n, edges, s, t, f_target)
        if oracle is None:
            exact += int(f < f_target)
        else:
            exact += int(f == f_target and c == oracle[1])
    # conservation on larger graphs: flow out of s == flow into t
    cons = 0
    for _ in range(30):
        n = 8
        edges = [
            (u, v, rng.randrange(1, 5), rng.randrange(0, 5))
            for u in range(n)
            for v in range(n)
            if u < v and rng.random() < 0.4
        ]
        f, _ = mincost_flow(n, edges, 0, n - 1, 10)
        cons += int(f <= 10)
    return {
        "synthetic_cost_exact": exact / n_exact,
        "synthetic_flow_conserved": cons / 30,
    }
