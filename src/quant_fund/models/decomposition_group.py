"""Frobenius order in cyclotomic fields (SYNTHETIC)."""

from __future__ import annotations

import math


def mult_order(p: int, n: int) -> int:
    """ord of p mod n in (Z/n)*; 0 if not coprime."""
    if math.gcd(p, n) != 1:
        return 0
    k, cur = 1, p % n
    while cur != 1:
        cur = (cur * p) % n
        k += 1
    return k


def inertial_degree(p: int, n: int) -> int:
    """Residue degree of p in Q(zeta_n) = ord_n(p)."""
    return mult_order(p, n)


def _bench_decomposition_group(seed: int = 0) -> float:
    checks = []
    # ord_7(2) = 3 (2^3 = 8 = 1 mod 7)
    checks.append(mult_order(2, 7) == 3)
    # ord_5(2) = 4
    checks.append(mult_order(2, 5) == 4)
    # ord_8(3) = 2
    checks.append(mult_order(3, 8) == 2)
    # p dividing n has no Frobenius (ramified)
    checks.append(mult_order(7, 7) == 0)
    # ord_n(1) = 1: prime splits completely
    checks.append(mult_order(1, 11) == 1)
    # Frob order divides phi(n): ord_11(2) = 10 | 10
    checks.append(10 % mult_order(2, 11) == 0)
    return float(sum(checks) / len(checks))


def bench_decomposition_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decomposition_group": _bench_decomposition_group(seed)}
