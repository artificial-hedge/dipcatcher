"""Quotient rings Z/nZ: units, CRT decomposition (SYNTHETIC)."""

from __future__ import annotations

import math


def units_zn(n: int) -> frozenset[int]:
    return frozenset(a for a in range(n) if math.gcd(a, n) == 1)


def is_field_zn(n: int) -> bool:
    return len(units_zn(n)) == n - 1


def crt_pair(n1: int, n2: int, a1: int, a2: int) -> int:
    """x ≡ a1 mod n1, x ≡ a2 mod n2 (coprime)."""
    for x in range(n1 * n2):
        if x % n1 == a1 and x % n2 == a2:
            return x
    raise ValueError("no solution")


def _bench_quotient_ring(seed: int = 0) -> float:
    checks = []
    checks.append(units_zn(6) == frozenset({1, 5}))
    checks.append(units_zn(7) == frozenset({1, 2, 3, 4, 5, 6}))
    checks.append(is_field_zn(7))
    checks.append(not is_field_zn(6))
    checks.append(crt_pair(3, 4, 2, 3) == 11)
    checks.append(crt_pair(3, 5, 1, 2) == 7)
    return float(sum(checks) / len(checks))


def bench_quotient_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quotient_ring": _bench_quotient_ring(seed)}
