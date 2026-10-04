"""Matroid intersection: common independent sets (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def common_independent(
    indep1: set[frozenset[int]], indep2: set[frozenset[int]]
) -> set[frozenset[int]]:
    return indep1 & indep2


def max_common(indep1: set[frozenset[int]], indep2: set[frozenset[int]]) -> int:
    """Largest common independent set size."""
    return max((len(i) for i in indep1 & indep2), default=0)


def rank_certifies(indep1: set[frozenset[int]], indep2: set[frozenset[int]], n: int) -> bool:
    """Edmonds: max |I in I1 cap I2| = min over A of r1(A) + r2(E\\A).
    Verify the min is >= max on the toy instance."""

    def rank(indep: set[frozenset[int]], s: set[int]) -> int:
        return max((len(i & s) for i in indep), default=0)

    mx = max_common(indep1, indep2)
    mn = n + 1
    for r in range(n + 1):
        for a in combinations(range(n), r):
            a_set = set(a)
            mn = min(mn, rank(indep1, a_set) + rank(indep2, set(range(n)) - a_set))
    return bool(mn >= mx)


def _bench_matroid_intersect(seed: int = 0) -> float:
    checks = []
    # colorful spanning tree on K4 edges colored {0,0,1,1,2,2}:
    # graphic matroid x partition matroid (at most one edge per color)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    colors = [0, 0, 1, 1, 2, 2]

    def is_forest(ss: frozenset[int]) -> bool:
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in ss:
            u, v = edges[e]
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
        return True

    n = 6
    all_sets = {frozenset(s) for r in range(n + 1) for s in combinations(range(n), r)}
    graphic = {i for i in all_sets if is_forest(i)}
    partition = {i for i in all_sets if len({colors[e] for e in i}) == len(i)}
    mx = max_common(graphic, partition)
    # 3 edges of distinct colors forming a tree exist iff colorful ST exists
    checks.append(mx == 3)
    checks.append(rank_certifies(graphic, partition, n))
    # recolor with a repeated color pattern making no colorful tree:
    colors2 = [0, 0, 0, 0, 1, 1]
    partition2 = {i for i in all_sets if len({colors2[e] for e in i}) == len(i)}
    checks.append(max_common(graphic, partition2) == 2)
    return float(sum(checks) / len(checks))


def bench_matroid_intersect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matroid_intersect": _bench_matroid_intersect(seed)}
