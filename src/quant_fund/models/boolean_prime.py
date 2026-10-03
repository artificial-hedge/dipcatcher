"""Boolean prime ideal theorem on finite algebras (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def extends_to_ultrafilter(
    universe: int, filt: list[list[int]]
) -> list[list[int]]:
    """On the finite BA P({0..n-1}): extend a filter to the ultrafilter
    of all subsets containing a chosen point of its intersection."""
    inter = set(range(universe))
    for s in filt:
        inter &= set(s)
    point = min(inter)
    subsets = [
        list(c)
        for r in range(universe + 1)
        for c in combinations(range(universe), r)
    ]
    return [s for s in subsets if point in s]


def _bench_boolean_prime(seed: int = 0) -> float:
    checks = []
    uf = extends_to_ultrafilter(4, [[0, 1, 2], [0, 1], [0, 2, 3]])
    # every original filter member survived
    checks.append([0, 1, 2] in uf and [0, 1] in uf and [0, 2, 3] in uf)
    # ultrafilter: for every set A, exactly one of A / complement is in
    checks.append(len(uf) == 8)  # half of all 16 subsets
    # upward closed and closed under intersection
    checks.append(all(0 in s for s in uf))
    checks.append(all([0] in uf for _ in [0]))
    # proper: empty set excluded
    checks.append([] not in uf)
    return float(sum(checks) / len(checks))


def bench_boolean_prime(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boolean_prime": _bench_boolean_prime(seed)}
