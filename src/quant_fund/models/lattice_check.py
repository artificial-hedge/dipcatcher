"""Finite lattices: join/meet existence, distributive/modular laws (SYNTHETIC)."""

from __future__ import annotations

import itertools


def ups(le: dict[tuple[int, int], bool], x: int, elems: set[int]) -> set[int]:
    return {y for y in elems if le.get((x, y), False)}


def downs(le: dict[tuple[int, int], bool], x: int, elems: set[int]) -> set[int]:
    return {y for y in elems if le.get((y, x), False)}


def join(le: dict[tuple[int, int], bool], x: int, y: int, elems: set[int]) -> int | None:
    ub = ups(le, x, elems) & ups(le, y, elems)
    for z in ub:
        if all(le.get((z, w), False) for w in ub):
            return z
    return None


def meet(le: dict[tuple[int, int], bool], x: int, y: int, elems: set[int]) -> int | None:
    lb = downs(le, x, elems) & downs(le, y, elems)
    for z in lb:
        if all(le.get((w, z), False) for w in lb):
            return z
    return None


def is_lattice(le: dict[tuple[int, int], bool], elems: set[int]) -> bool:
    return all(
        join(le, x, y, elems) is not None and meet(le, x, y, elems) is not None
        for x, y in itertools.product(elems, repeat=2)
    )


def distributive(le: dict[tuple[int, int], bool], elems: set[int]) -> bool:
    for x, y, z in itertools.product(elems, repeat=3):
        lhs = (
            meet(le, x, join(le, y, z, elems) or 0, elems)
            if join(le, y, z, elems) is not None
            else None
        )
        rhs = (
            join(le, meet(le, x, y, elems) or 0, meet(le, x, z, elems) or 0, elems)
            if (meet(le, x, y, elems) is not None and meet(le, x, z, elems) is not None)
            else None
        )
        if lhs != rhs:
            return False
    return True


def _bench_lattice_check(seed: int = 0) -> float:
    checks = []
    # divisibility lattice on {1,2,3,6}: LCM/GCD
    elems = {1, 2, 3, 6}
    le = {(a, b): b % a == 0 for a in elems for b in elems}
    checks.append(is_lattice(le, elems))
    checks.append(join(le, 2, 3, elems) == 6)
    checks.append(meet(le, 2, 6, elems) == 2)
    checks.append(distributive(le, elems))
    # pentagon N5 is a lattice but NOT distributive: elements {0=bot,1=top, a,b chain a<b, c atom}
    elems5 = {0, 1, 2, 3, 4}
    le5 = {(x, x): True for x in elems5}
    for x in elems5:
        le5[(0, x)] = True
        le5[(x, 1)] = True
    le5[(2, 3)] = True  # chain a=2 < b=3
    checks.append(is_lattice(le5, elems5))
    checks.append(not distributive(le5, elems5))
    return float(sum(checks) / len(checks))


def bench_lattice_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lattice_check": _bench_lattice_check(seed)}
