"""Graph minors: K3-minor-free iff forest; small topological checks (SYNTHETIC)."""

from __future__ import annotations


def has_cycle(adj: list[set[int]]) -> bool:
    n = len(adj)
    color = [0] * n

    def dfs(u: int, parent: int) -> bool:
        color[u] = 1
        for w in adj[u]:
            if color[w] == 1 and w != parent:
                return True
            if color[w] == 0 and dfs(w, u):
                return True
        color[u] = 2
        return False

    return any(color[i] == 0 and dfs(i, -1) for i in range(n))


def has_k3_minor(adj: list[set[int]]) -> bool:
    """K3 is a minor iff the graph contains a cycle."""
    return has_cycle(adj)


def has_k4_minor(adj: list[set[int]]) -> bool:
    """Small-instance K4-minor check via exhaustive edge-contraction.

    K4 is a minor iff some contraction sequence yields a 4-vertex supergraph
    of K4's structure; for small graphs we test whether some subgraph induced
    by 4 connected branch sets gives all six cross-edges. We search partitions
    of a spanning connected subgraph into 4 parts (n small only).
    """
    n = len(adj)
    if n < 4:
        return False
    # branch sets: need 4 disjoint connected subsets with all 6 links.
    # Enumerate by choosing 4 "roots" then greedily expanding is hard; for
    # n <= 12 do DFS over assignments of each vertex to a part or none.
    from itertools import product

    def connected(part_verts: set[int]) -> bool:
        if not part_verts:
            return False
        seen = set()
        stack = [next(iter(part_verts))]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] & part_verts)
        return seen == part_verts

    verts = list(range(n))
    for assign in product(range(5), repeat=n):
        parts: list[set[int]] = [set() for _ in range(4)]
        for v, a in zip(verts, assign, strict=True):
            if a < 4:
                parts[a].add(v)
        if not all(connected(p) for p in parts):
            continue
        ok = True
        for i in range(4):
            for j in range(i + 1, 4):
                if not any(b in adj[a] for a in parts[i] for b in parts[j]):
                    ok = False
        if ok:
            return True
    return False


def _mk(edges: list[tuple[int, int]], n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _bench_graph_minor(seed: int = 0) -> float:
    checks = []
    tree = _mk([(0, 1), (1, 2), (1, 3), (0, 4)], 5)
    checks.append(not has_k3_minor(tree))
    c4 = _mk([(i, (i + 1) % 4) for i in range(4)], 4)
    checks.append(has_k3_minor(c4))
    k4 = _mk([(a, b) for a in range(4) for b in range(a + 1, 4)], 4)
    checks.append(has_k4_minor(k4))
    # K_{3,3} has K5? No — but it does have K4 minor? K3,3 has no K4 minor
    # actually K_{3,3} DOES contain a K4 minor? Contract... K_{3,3} is
    # non-planar but its Hadwiger number is 3? It contains a K4 minor iff
    # it has treewidth >= 3... K_{3,3} does have K4 minor? K_{3,3} has no K4
    # minor because contracting gives at most K_{2,3}+... Use series-parallel
    # instead: diamond (K4 minus edge) has K4? No - series-parallel graphs
    # are exactly K4-minor-free. Diamond IS series-parallel -> no K4 minor.
    diamond = _mk([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)], 4)
    checks.append(not has_k4_minor(diamond))
    # wheel W5 has K4 minor (hub + triangle of rim)
    w5 = _mk([(i, (i + 1) % 4) for i in range(4)] + [(4, i) for i in range(4)], 5)
    checks.append(has_k4_minor(w5))
    return float(sum(checks) / len(checks))


def bench_graph_minor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_graph_minor": _bench_graph_minor(seed)}
