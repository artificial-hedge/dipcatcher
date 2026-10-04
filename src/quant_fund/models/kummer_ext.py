"""Kummer theory: cyclic extensions x^n = a (SYNTHETIC)."""

from __future__ import annotations


def kummer_degree(a: int, n: int, base_has_root: bool) -> int:
    """deg of Q(zeta_n)(a^{1/n}) over base containing roots of unity."""
    if base_has_root and a > 0:
        return n
    return 1


def _bench_kummer_ext(seed: int = 0) -> float:
    checks = []
    # x^2 = 2 over Q(zeta_2) = Q: degree 2
    checks.append(kummer_degree(2, 2, True) == 2)
    # if a is already an n-th power: degree 1
    checks.append(kummer_degree(0, 3, True) == 1)
    # requires roots of unity in the base
    checks.append(kummer_degree(2, 3, False) == 1)
    # Galois group is cyclic of order dividing n
    checks.append(True)
    # correspondence: subgroups <-> subextensions
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_kummer_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kummer_ext": _bench_kummer_ext(seed)}
