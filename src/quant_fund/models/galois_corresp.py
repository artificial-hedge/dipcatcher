"""Galois correspondence on subgroups of Z/n vs divisors of n (SYNTHETIC)."""

from __future__ import annotations

import math


def subgroups_zn(n: int) -> list[int]:
    """Subgroups of Z/n are <d> for d | n; represented by step size d."""
    return [d for d in range(1, n + 1) if n % d == 0]


def _bench_galois_corresp(seed: int = 0) -> float:
    checks = []
    # subgroups of Z/12: divisors 1,2,3,4,6,12 -> 6 subgroups
    checks.append(subgroups_zn(12) == [1, 2, 3, 4, 6, 12])
    # Z/p (prime) has only trivial subgroups
    checks.append(subgroups_zn(7) == [1, 7])
    # order of <d> in Z/n is n/d
    checks.append(all(12 // d == len(range(0, 12, d)) for d in subgroups_zn(12)))
    # lattice anti-inclusion: <a> <= <b> iff b | a (in Z/n as ideals)
    checks.append(set(range(0, 12, 6)) <= set(range(0, 12, 3)))  # <6> < <3> since 3|6
    # Galois group of Q(zeta_n) ~ (Z/n)^x: order phi(n)
    phi = sum(1 for k in range(1, 13) if math.gcd(k, 12) == 1)
    checks.append(phi == 4)
    # (Z/12)^x = {1,5,7,11}, each element order 2 (Klein)
    units = [k for k in range(1, 13) if math.gcd(k, 12) == 1]
    checks.append(all((u * u) % 12 == 1 for u in units))
    return float(sum(checks) / len(checks))


def bench_galois_corresp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_corresp": _bench_galois_corresp(seed)}
