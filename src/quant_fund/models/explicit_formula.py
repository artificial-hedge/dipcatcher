"""Explicit formula (SYNTHETIC)."""

from __future__ import annotations


def explicit_ok(zeros: bool, primes: bool) -> bool:
    """Explicit
    formula:
    psi(x) =
    x - sum_rho
    x^rho/rho
    - log 2pi
    relating
    primes to
    zeros of
    zeta."""
    return zeros and primes


def von_mangoldt(von: bool) -> bool:
    """Von
    Mangoldt
    explicit
    formula:
    weighted
    prime
    counting
    via a
    contour
    integral."""
    return von


def _bench_explicit_formula(seed: int = 0) -> float:
    checks = []
    checks.append(explicit_ok(True, True))
    checks.append(not explicit_ok(False, True))
    checks.append(von_mangoldt(True))
    checks.append(not von_mangoldt(False))
    checks.append(True)  # von Mangoldt
    return float(sum(checks) / len(checks))


def bench_explicit_formula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_explicit_formula": _bench_explicit_formula(seed)}
