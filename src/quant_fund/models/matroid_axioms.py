"""Matroid axiom verification on cycle matroids (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def is_forest(edges: set[int], edge_list: list[tuple[int, int]]) -> bool:
    """Subset of edges forms a forest iff no cycle: union-find."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        u, v = edge_list[e]
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def powerset_independent(
    indep: set[frozenset[int]],
) -> bool:
    """Independence axiom: subsets of independent sets are independent."""
    return all(
        frozenset(sub) in indep
        for i in indep
        for r in range(len(i) + 1)
        for sub in combinations(i, r)
    )


def augmentation(indep: set[frozenset[int]]) -> bool:
    """Augmentation axiom: |I|<|J| => exists e in J\\I with I+e independent."""
    for i in indep:
        for j in indep:
            if len(i) < len(j) and not any(frozenset(set(i) | {e}) in indep for e in j - i):
                return False
    return True


def _bench_matroid_axioms(seed: int = 0) -> float:
    checks = []
    # cycle matroid of K4 (4 vertices, 6 edges)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n = 6
    indep = {
        frozenset(s)
        for r in range(n + 1)
        for s in combinations(range(n), r)
        if is_forest(set(s), edges)
    }
    checks.append(powerset_independent(indep))
    checks.append(augmentation(indep))
    # rank = n_vertices - components = 3 for connected K4
    bases = [i for i in indep if len(i) == 3]
    checks.append(len(bases) == 16)  # K4 has 16 spanning trees (Cayley 4^2)
    # the triangle {0,1,3} = edges (0,1),(0,2),(1,2) is a circuit
    checks.append(frozenset({0, 1, 3}) not in indep)
    checks.append(frozenset({0, 1}) in indep)
    return float(sum(checks) / len(checks))


def bench_matroid_axioms(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_axioms": _bench_matroid_axioms(seed)}
