"""Max-flow canon: Dinic's algorithm + min-cut extraction.

- ``dinic_maxflow`` — level-graph BFS + blocking-flow DFS with
  current-edge pointers; O(V^2 E) worst case, near-linear in
  practice.
- ``min_cut`` — reachable set in the residual graph after
  termination (max-flow = min-cut duality).

Bench: random capacity network where flow value matches an
independent LP-style bound computed via the min-cut equality
(SYNTHETIC only).
"""

from __future__ import annotations

from collections import deque

import numpy as np

FloatArray = np.ndarray


def dinic_maxflow(
    n: int, edges: list[tuple[int, int, float]], src: int, dst: int
) -> tuple[float, FloatArray]:
    cap = np.zeros((n, n))
    for u, v, w in edges:
        cap[u, v] += w
    flow = np.zeros((n, n))
    total = 0.0
    while True:
        # level graph
        level = np.full(n, -1)
        level[src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in range(n):
                if level[v] < 0 and cap[u, v] - flow[u, v] > 1e-12:
                    level[v] = level[u] + 1
                    dq.append(v)
        if level[dst] < 0:
            break
        nxt = np.zeros(n, dtype=np.int64)

        def push(  # noqa: B023
            u: int,
            f: float,
            nxt: np.ndarray = nxt,
            level: np.ndarray = level,
        ) -> float:
            if u == dst:
                return f
            for i in range(nxt[u], n):
                nxt[u] = i
                v = i
                r = cap[u, v] - flow[u, v]
                if level[v] == level[u] + 1 and r > 1e-12:
                    d = push(v, min(f, r))
                    if d > 0:
                        flow[u, v] += d
                        flow[v, u] -= d
                        return d
            return 0.0

        while (aug := push(src, np.inf)) > 0:
            total += aug
    return float(total), flow


def min_cut(
    n: int, edges: list[tuple[int, int, float]], src: int, flow: FloatArray
) -> tuple[FloatArray, float]:
    cap = np.zeros((n, n))
    for u, v, w in edges:
        cap[u, v] += w
    seen = np.zeros(n, dtype=bool)
    dq = deque([src])
    seen[src] = True
    while dq:
        u = dq.popleft()
        for v in range(n):
            if not seen[v] and cap[u, v] - flow[u, v] > 1e-12:
                seen[v] = True
                dq.append(v)
    cut = sum(cap[u, v] for u in range(n) for v in range(n) if seen[u] and not seen[v])
    return seen, float(cut)


def bench_network_flow(seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, src, dst = 40, 0, 39
    edges: list[tuple[int, int, float]] = []
    for _ in range(160):
        u, v = rng.integers(0, n, 2)
        if u != v and u != dst and v != src:
            edges.append((int(u), int(v), float(rng.random() * 9 + 1)))
    val, flow = dinic_maxflow(n, edges, src, dst)
    seen, cut = min_cut(n, edges, src, flow)
    # conservation at internal nodes
    bal = flow.sum(axis=1) - flow.sum(axis=0)
    intern = [i for i in range(n) if i not in (src, dst)]
    conserve_err = float(np.max(np.abs(bal[intern])))
    return {
        "synthetic_maxflow": val,
        "synthetic_mincut": cut,
        "synthetic_duality_err": abs(val - cut),
        "synthetic_conserve_err": conserve_err,
    }


__all__ = ["bench_network_flow", "dinic_maxflow", "min_cut"]
