"""Cyclotomic fields: degrees and primitive roots (SYNTHETIC)."""

from __future__ import annotations

import math


def cyclotomic_degree(n: int) -> int:
    """[Q(zeta_n):Q] = phi(n)."""
    return sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1)


def primitive_roots(n: int) -> list[int]:
    """Primitive n-th roots of unity = powers coprime to n."""
    return [k for k in range(1, n + 1) if math.gcd(k, n) == 1]


def _bench_cyclotomic_field(seed: int = 0) -> float:
    checks = []
    checks.append(cyclotomic_degree(5) == 4)
    checks.append(cyclotomic_degree(8) == 4)
    checks.append(cyclotomic_degree(12) == 4)
    checks.append(cyclotomic_degree(7) == 6)
    # primitive 5th roots = {1,2,3,4}
    checks.append(primitive_roots(5) == [1, 2, 3, 4])
    # phi multiplicative: phi(15) = phi(3)*phi(5) = 8
    checks.append(cyclotomic_degree(15) == 8)
    return float(sum(checks) / len(checks))


def bench_cyclotomic_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclotomic_field": _bench_cyclotomic_field(seed)}
