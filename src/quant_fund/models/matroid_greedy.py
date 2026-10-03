"""Graphic matroid greedy = Kruskal MST, verified against brute force (SYNTHETIC)."""

from __future__ import annotations

Edge = tuple[int, int, float]  # (u, v, w)


def acyclic(edges: list[Edge], cand: Edge) -> bool:
    """Union-find acyclicity test."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    all_e = edges + [cand]
    verts = set()
    for u, v, _ in all_e:
        verts.add(u)
        verts.add(v)
    for u, v, _ in all_e:
        pu, pv = find(u), find(v)
        if pu == pv:
            return False
        parent[pu] = pv
    return True


def greedy_basis(edges: list[Edge]) -> list[Edge]:
    """Matroid greedy: sort by weight, add if independent."""
    basis: list[Edge] = []
    for e in sorted(edges, key=lambda x: x[2]):
        if acyclic(basis, e):
            basis.append(e)
    return basis


def mst_brute(edges: list[Edge], n: int) -> float:
    """Minimum spanning set weight via exhaustive subsets."""
    import itertools

    best = float("inf")
    for r in range(1, len(edges) + 1):
        for combo in itertools.combinations(edges, r):
            verts = set()
            for u, v, _ in combo:
                verts.add(u)
                verts.add(v)
            # connected?
            adj: dict[int, list[int]] = {}
            for u, v, _ in combo:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            seen = set()
            stack = [next(iter(verts))] if verts else []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(adj.get(x, []))
            if seen == verts and all(u in seen for u, v, _ in combo):
                w = sum(e[2] for e in combo)
                if len(verts) == n and w < best:
                    best = w
    return best


def _bench_matroid_greedy(seed: int = 0) -> float:
    checks = []
    edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 1.0), (0, 3, 5.0)]
    basis = greedy_basis(edges)
    checks.append(sum(e[2] for e in basis) == 4.0)
    checks.append(len(basis) == 3)
    checks.append(basis[0] == (0, 1, 1.0))
    # triangle: basis drops heaviest edge
    tri = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)]
    checks.append(sum(e[2] for e in greedy_basis(tri)) == 2.0)
    # greedy matches brute-force MST on 4-node instance
    checks.append(mst_brute(edges, 4) == 4.0)
    # acyclic predicate
    checks.append(acyclic([(0, 1, 1.0)], (1, 2, 1.0)))
    checks.append(not acyclic([(0, 1, 1.0), (1, 2, 1.0)], (0, 2, 1.0)))
    return float(sum(checks) / len(checks))


def bench_matroid_greedy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_greedy": _bench_matroid_greedy(seed)}
