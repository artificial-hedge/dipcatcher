"""Graph-decomposition canon: Tarjan SCC, bridges, articulation
points.

- ``tarjan_scc`` — iterative Tarjan strongly-connected-components
  (lowlink, on-stack set), returns component labels + sizes.
- ``bridges`` — undirected bridges via DFS low.
- ``articulation`` — undirected cut-vertices via DFS low +
  child-count root rule.

Bench: planted SCC structure (two cycles + linking path) and a
bridge/articulation tree (SYNTHETIC only).
"""

from __future__ import annotations

import numpy as np

IntArray = np.ndarray


def tarjan_scc(n: int, edges: list[tuple[int, int]]) -> tuple[IntArray, int]:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
    idx = np.full(n, -1)
    low = np.zeros(n, dtype=np.int64)
    onstack = np.zeros(n, dtype=bool)
    comp = np.full(n, -1)
    stack: list[int] = []
    n_comp = 0
    counter = 0
    for s in range(n):
        if idx[s] >= 0:
            continue
        work = [(s, 0)]
        while work:
            u, ci = work[-1]
            if ci == 0:
                idx[u] = low[u] = counter
                counter += 1
                stack.append(u)
                onstack[u] = True
            recursed = False
            for i in range(ci, len(g[u])):
                v = g[u][i]
                work[-1] = (u, i + 1)
                if idx[v] < 0:
                    work.append((v, 0))
                    recursed = True
                    break
                elif onstack[v]:
                    low[u] = min(low[u], idx[v])
            if recursed:
                continue
            work.pop()
            if work:
                p = work[-1][0]
                low[p] = min(low[p], low[u])
            if low[u] == idx[u]:
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    comp[w] = n_comp
                    if w == u:
                        break
                n_comp += 1
    return comp, n_comp


def bridges(n: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    idx = np.full(n, -1)
    low = np.zeros(n, dtype=np.int64)
    out: list[tuple[int, int]] = []
    counter = 0
    for s in range(n):
        if idx[s] >= 0:
            continue
        work = [(s, -1, 0)]  # node, parent, child-iter
        while work:
            u, p, ci = work[-1]
            if ci == 0:
                idx[u] = low[u] = counter
                counter += 1
            recursed = False
            for i in range(ci, len(g[u])):
                v = g[u][i]
                work[-1] = (u, p, i + 1)
                if v == p and i == ci:
                    continue
                if v == p:
                    continue
                if idx[v] < 0:
                    work.append((v, u, 0))
                    recursed = True
                    break
                else:
                    low[u] = min(low[u], idx[v])
            if recursed:
                continue
            work.pop()
            if work:
                p2 = work[-1][0]
                low[p2] = min(low[p2], low[u])
                if low[u] > idx[p2]:
                    out.append((min(p2, u), max(p2, u)))
    return out


def articulation(n: int, edges: list[tuple[int, int]]) -> IntArray:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    idx = np.full(n, -1)
    low = np.zeros(n, dtype=np.int64)
    is_ap = np.zeros(n, dtype=bool)
    counter = 0
    for s in range(n):
        if idx[s] >= 0:
            continue
        work = [(s, -1, 0)]
        children: dict[int, int] = {}
        while work:
            u, p, ci = work[-1]
            if ci == 0:
                idx[u] = low[u] = counter
                counter += 1
                children[u] = 0
            recursed = False
            for i in range(ci, len(g[u])):
                v = g[u][i]
                work[-1] = (u, p, i + 1)
                if v == p:
                    continue
                if idx[v] < 0:
                    children[u] += 1
                    work.append((v, u, 0))
                    recursed = True
                    break
                else:
                    low[u] = min(low[u], idx[v])
            if recursed:
                continue
            work.pop()
            if work:
                p2 = work[-1][0]
                low[p2] = min(low[p2], low[u])
                if p != -1 and low[u] >= idx[p2]:
                    is_ap[p2] = True
            elif children.get(s, 0) > 1:
                is_ap[s] = True
    return is_ap


def bench_graph_components(seed: int = 0) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    # Two 5-cycles joined by a one-way link -> 2 SCCs (plus a tail).
    edges = (
        [(i, (i + 1) % 5) for i in range(5)]
        + [(5 + i, 5 + (i + 1) % 5) for i in range(5)]
        + [(0, 5), (5, 10)]
    )
    comp, ncomp = tarjan_scc(11, edges)
    sizes = np.bincount(comp)
    big = int((sizes == 5).sum())
    # Bridge tree: path 0-1-2 with leaves 3,4 on node 1.
    bedges = [(0, 1), (1, 2), (1, 3), (1, 4)]
    br = bridges(5, bedges)
    ap = articulation(5, bedges)
    return {
        "synthetic_scc_count": float(ncomp),
        "synthetic_scc_big": float(big),
        "synthetic_bridge_count": float(len(br)),
        "synthetic_articulation": float(ap[1]),
    }


__all__ = [
    "articulation",
    "bench_graph_components",
    "bridges",
    "tarjan_scc",
]
