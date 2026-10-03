"""Incidence algebra: zeta and Mobius functions (SYNTHETIC)."""

from __future__ import annotations

import functools


def mobius(leq, elems: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """mu(x, z) computed by mu(x,x)=1 and mu(x,z) = -sum_{x<=y<z} mu(x,y)."""
    mu: dict[tuple[int, int], int] = {}
    for x in elems:
        mu[(x, x)] = 1
    # longest chains first: iterate pairs by length via recursive memo

    @functools.cache
    def m(x: int, z: int) -> int:
        if x == z:
            return 1
        if not leq(x, z):
            return 0
        s = 0
        for y in elems:
            if leq(x, y) and leq(y, z) and y != z:
                s += m(x, y)
        return -s

    for x in elems:
        for z in elems:
            mu[(x, z)] = m(x, z)
    return mu


def _bench_zeta_mobius(seed: int = 0) -> float:
    checks = []
    def leq(a: int, b: int) -> bool:
        return a <= b

    elems = (0, 1, 2)
    mu = mobius(leq, elems)
    # chain: mu alternates -1 on covers, 0 elsewhere
    checks.append(mu[(0, 1)] == -1 and mu[(1, 2)] == -1)
    checks.append(mu[(0, 2)] == 0)
    # mu* zeta = delta: sum_y mu(x,y) for y<=z gives 0 unless x==z
    s = sum(mu[(0, y)] for y in elems if leq(y, 2) and leq(0, y))
    checks.append(s == 0)
    s2 = sum(mu[(0, y)] for y in elems if leq(y, 0))
    checks.append(s2 == 1)
    # boolean lattice B2: mu(bottom, top) = +1
    def leq2(a: int, b: int) -> bool:
        return (a & b) == a

    b2 = tuple(range(4))
    mu2 = mobius(leq2, b2)
    checks.append(mu2[(0, 3)] == 1)
    return float(sum(checks) / len(checks))


def bench_zeta_mobius(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zeta_mobius": _bench_zeta_mobius(seed)}
