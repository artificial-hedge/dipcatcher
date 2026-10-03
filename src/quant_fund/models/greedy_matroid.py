"""Greedy is optimal on matroids (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def greedy_basis(weights: list[float], indep: set[frozenset[int]]) -> frozenset[int]:
    """Max-weight basis by descending greedy."""
    order = sorted(range(len(weights)), key=lambda i: -weights[i])
    b: set[int] = set()
    for e in order:
        if frozenset(b | {e}) in indep:
            b.add(e)
    return frozenset(b)


def brute_max(weights: list[float], indep: set[frozenset[int]]) -> float:
    best = -1.0
    for i in indep:
        w = sum(weights[e] for e in i)
        if w > best and not any(i < j and j in indep for j in indep if len(j) == len(i) + 1):
            best = w
    # restrict to maximal sets
    maximals = [i for i in indep if not any(i < j and j in indep for j in indep)]
    return max(sum(weights[e] for e in i) for i in maximals)


def _bench_greedy_matroid(seed: int = 0) -> float:
    checks = []
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def is_forest(ss: set[int]) -> bool:
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
    indep = {
        frozenset(s) for r in range(n + 1) for s in combinations(range(n), r) if is_forest(set(s))
    }
    # several weight profiles: greedy weight == brute-force max basis weight
    for w in ([5.0, 1, 4, 2, 3, 0.5], [1.0, 1, 1, 1, 1, 1], [3.0, 3, 3, 1, 2, 2]):
        g = greedy_basis(list(w), indep)
        checks.append(abs(sum(w[e] for e in g) - brute_max(list(w), indep)) < 1e-9)
    # on a NON-matroid family greedy can fail: family {0},{1,2} maximal
    # independent both size matters — the augmentation fails, and greedy
    # with weights [0.4, 0.3, 0.3] picks {0} weight 0.4 < {1,2} weight 0.6
    bad_family = {frozenset(), frozenset({0}), frozenset({1}), frozenset({2}), frozenset({1, 2})}
    g = greedy_basis([0.4, 0.3, 0.3], bad_family)
    checks.append(g == frozenset({0}))
    checks.append(brute_max([0.4, 0.3, 0.3], bad_family) > sum([0.4, 0.3, 0.3][e] for e in g))
    return float(sum(checks) / len(checks))


def bench_greedy_matroid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_greedy_matroid": _bench_greedy_matroid(seed)}
