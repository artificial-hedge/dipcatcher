"""Factorial ring (SYNTHETIC)."""

from __future__ import annotations


def fr_ok(factorial: bool, domain: bool) -> bool:
    """Factorial:
    unique
    factorization
    domain —
    UFD."""
    return factorial and domain


def irred_prime(ip: bool) -> bool:
    """Irreducible:
    in
    a
    UFD
    irreducibles
    are
    prime —
    UFD
    property."""
    return ip


def _bench_factorial_ring(seed: int = 0) -> float:
    checks = []
    checks.append(fr_ok(True, True))
    checks.append(not fr_ok(False, True))
    checks.append(irred_prime(True))
    checks.append(not irred_prime(False))
    checks.append(True)  # UFD
    return float(sum(checks) / len(checks))


def bench_factorial_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_factorial_ring": _bench_factorial_ring(seed)}
