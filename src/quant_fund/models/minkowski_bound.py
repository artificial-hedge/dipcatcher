"""Minkowski bound: every ideal class has a small ideal (SYNTHETIC)."""

from __future__ import annotations


def minkowski_c(r1: int, r2: int, n: int, disc: int) -> float:
    """M_K = (4/pi)^{r2} * n! / n^n * sqrt|disc|."""
    from math import factorial, pi, sqrt

    return float((4.0 / pi) ** r2 * factorial(n) / n**n * sqrt(abs(disc)))


def _bench_minkowski_bound(seed: int = 0) -> float:
    checks = []
    # imaginary quadratic disc -3: bound < 2 -> h = 1
    checks.append(minkowski_c(0, 1, 2, -3) < 2.0)
    # disc -15: bound ~ 2.47 -> ideals of norm <= 2
    checks.append(minkowski_c(0, 1, 2, -15) < 3.0)
    # real quadratic disc 8 has a larger bound than disc 5
    checks.append(minkowski_c(2, 0, 2, 8) > minkowski_c(2, 0, 2, 5))
    # disc -163 (largest Heegner): bound stays finite and below 9
    checks.append(minkowski_c(0, 1, 2, -163) < 9.0)
    # and exceeds the disc -15 bound since it scales with sqrt|disc|
    checks.append(minkowski_c(0, 1, 2, -163) > minkowski_c(0, 1, 2, -15))
    if sum(checks) != len(checks):
        raise ValueError("Minkowski-bound oracle checks failed")
    return float(sum(checks) / len(checks))


def bench_minkowski_bound(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minkowski_bound": _bench_minkowski_bound(seed)}
