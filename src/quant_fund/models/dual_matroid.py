"""Dual matroid: bases are complements of bases (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def uniform_indep(k: int, n: int) -> set[frozenset[int]]:
    return {frozenset(s) for r in range(k + 1) for s in combinations(range(n), r)}


def bases(indep: set[frozenset[int]]) -> set[frozenset[int]]:
    return {i for i in indep if not any(i < j and j in indep for j in indep)}


def dual_bases(indep: set[frozenset[int]], n: int) -> set[frozenset[int]]:
    ground = frozenset(range(n))
    return {ground - b for b in bases(indep)}


def dual_rank(indep: set[frozenset[int]], n: int, a: set[int]) -> int:
    """r*(A) = |A| + r(E\\A) - r(E)."""

    def rank(s: set[int]) -> int:
        return max((len(i & s) for i in indep), default=0)

    ground = set(range(n))
    return len(a) + rank(ground - a) - rank(ground)


def _bench_dual_matroid(seed: int = 0) -> float:
    checks = []
    u23 = uniform_indep(2, 3)
    # dual of U_{2,3} is U_{1,3}: bases {0,1},{0,2},{1,2} -> complements
    checks.append(dual_bases(u23, 3) == bases(uniform_indep(1, 3)))
    u25 = uniform_indep(2, 5)
    checks.append(dual_bases(u25, 5) == bases(uniform_indep(3, 5)))
    # dual of dual = original
    d = dual_bases(u23, 3)
    dd = {frozenset(range(3)) - b for b in d}
    checks.append(dd == bases(u23))
    # rank formula check: r*({0}) on U_{2,3}: 1 + r({1,2}) - 2 = 1+2-2 = 1
    checks.append(dual_rank(u23, 3, {0}) == 1)
    # r*({0,1,2}) = 3 + 0 - 2 = 1 = rank of dual U_{1,3}
    checks.append(dual_rank(u23, 3, {0, 1, 2}) == 1)
    return float(sum(checks) / len(checks))


def bench_dual_matroid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dual_matroid": _bench_dual_matroid(seed)}
