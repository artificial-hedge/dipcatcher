"""Subobject classifier and subobject count in Set (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def subobjects(n: int) -> int:
    """Number of subobjects of an n-element set = 2^n."""
    return int(2**n)


def characteristic_map(sub: list[int], univ: int) -> list[bool]:
    """chi_A: U -> Omega = {true, false}."""
    return [i in sub for i in range(univ)]


def _bench_topos_subobj(seed: int = 0) -> float:
    checks = []
    checks.append(subobjects(2) == 4)
    checks.append(subobjects(3) == 8)
    # characteristic maps are bijective with subobjects
    univ = 3
    subs = [list(c) for r in range(univ + 1) for c in combinations(range(univ), r)]
    checks.append(len(subs) == 8)
    chi = characteristic_map([0, 2], 3)
    checks.append(chi == [True, False, True])
    # pullback of true: 1 -> Omega along chi gives back the subobject
    checks.append([i for i in range(3) if chi[i]] == [0, 2])
    return float(sum(checks) / len(checks))


def bench_topos_subobj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_topos_subobj": _bench_topos_subobj(seed)}
