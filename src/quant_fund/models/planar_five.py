"""Six-color theorem via degeneracy ordering for planar graphs (SYNTHETIC)."""

from __future__ import annotations


def degeneracy_color(adj: list[set[int]], k: int = 6) -> list[int] | None:
    """Order vertices by repeatedly removing a min-degree vertex (<= k-1... use <=5).

    Planar graphs have min degree <= 5, so this ordering always exists and
    greedy reinsertion colors with <= 6 colors.
    """
    n = len(adj)
    deg = [len(a) for a in adj]
    removed = [False] * n
    order: list[int] = []
    for _ in range(n):
        v = min((i for i in range(n) if not removed[i]), key=lambda i: deg[i])
        if deg[v] > k - 1:
            return None  # not 5-degenerate here
        removed[v] = True
        order.append(v)
        for u in adj[v]:
            deg[u] -= 1
    color = [-1] * n
    for v in reversed(order):
        used = {color[u] for u in adj[v] if color[u] != -1}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def valid_coloring(adj: list[set[int]], color: list[int]) -> bool:
    return all(color[u] != color[v] for u in range(len(adj)) for v in adj[u] if v > u)


def _mk(edges: list[tuple[int, int]], n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _icosahedron() -> list[set[int]]:
    """12-vertex icosahedron: two pentagonal pyramids + antiprism."""
    edges = [(i, (i + 1) % 5) for i in range(5)]
    edges += [(i + 5, (i + 1) % 5 + 5) for i in range(5)]
    edges += [(i, i + 5) for i in range(5)]
    edges += [(i, (i + 1) % 5 + 5) for i in range(5)]
    edges += [(10, i) for i in range(5)]
    edges += [(11, i + 5) for i in range(5)]
    return _mk(edges, 12)


def _bench_planar_five(seed: int = 0) -> float:
    checks = []
    ico = _icosahedron()
    c = degeneracy_color(ico)
    checks.append(c is not None and valid_coloring(ico, c))
    checks.append(c is not None and max(c) + 1 <= 6)
    # wheel W8 (planar): rim + hub
    w = _mk([(i, (i + 1) % 7) for i in range(7)] + [(7, i) for i in range(7)], 8)
    cw = degeneracy_color(w)
    checks.append(cw is not None and valid_coloring(w, cw))
    # 3x4 grid (planar, bipartite)
    grid = _mk(
        [(i * 4 + j, i * 4 + j + 1) for i in range(3) for j in range(3)]
        + [(i * 4 + j, (i + 1) * 4 + j) for i in range(2) for j in range(4)],
        12,
    )
    cg = degeneracy_color(grid)
    checks.append(cg is not None and valid_coloring(grid, cg))
    checks.append(cg is not None and max(cg) + 1 <= 2 + 1)
    # K4 (planar, needs 4): colored validly
    k4 = _mk([(a, b) for a in range(4) for b in range(a + 1, 4)], 4)
    ck = degeneracy_color(k4)
    checks.append(ck is not None and valid_coloring(k4, ck) and max(ck) + 1 == 4)
    return float(sum(checks) / len(checks))


def bench_planar_five(seed: int = 0) -> dict[str, float]:
    return {"synthetic_planar_five": _bench_planar_five(seed)}
