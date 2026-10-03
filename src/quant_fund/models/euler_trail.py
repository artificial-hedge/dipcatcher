"""Eulerian trails: Hierholzer's algorithm + degree/parity conditions (SYNTHETIC)."""

from __future__ import annotations

Graph = dict[int, list[int]]  # multigraph adjacency with multiplicity


def odd_degree_count(g: Graph) -> int:
    return sum(1 for v in g if len(g[v]) % 2 == 1)


def euler_status(g: Graph) -> str:
    """'circuit' | 'trail' | 'none' for connected multigraphs."""
    odd = odd_degree_count(g)
    if odd == 0:
        return "circuit"
    if odd == 2:
        return "trail"
    return "none"


def hierholzer(g: Graph, start: int) -> list[int]:
    """Eulerian circuit edge-vertex sequence."""
    adj = {v: list(ns) for v, ns in g.items()}
    stack = [start]
    path = []
    while stack:
        v = stack[-1]
        if adj[v]:
            w = adj[v].pop()
            adj[w].remove(v)
            stack.append(w)
        else:
            path.append(stack.pop())
    return path[::-1]


def _bench_euler_trail(seed: int = 0) -> float:
    checks = []
    # triangle: circuit
    tri = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    checks.append(euler_status(tri) == "circuit")
    circ = hierholzer(tri, 0)
    checks.append(circ[0] == circ[-1] and len(circ) == 4)
    # path: trail (2 odd vertices)
    path = {0: [1], 1: [0, 2], 2: [1]}
    checks.append(euler_status(path) == "trail")
    # star: 3 odd -> none
    star = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    checks.append(euler_status(star) == "none")
    # K4: all odd degree 3 (4 odd) -> none
    k4 = {i: [j for j in range(4) if j != i] for i in range(4)}
    checks.append(euler_status(k4) == "none")
    # square: circuit
    sq = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    checks.append(euler_status(sq) == "circuit")
    return float(sum(checks) / len(checks))


def bench_euler_trail(seed: int = 0) -> dict[str, float]:
    return {"synthetic_euler_trail": _bench_euler_trail(seed)}
