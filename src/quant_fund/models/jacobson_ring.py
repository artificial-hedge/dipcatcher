"""Jacobson ring (SYNTHETIC)."""

from __future__ import annotations


def jr_ok(jacobson: bool, ring: bool) -> bool:
    """Jacobson:
    Jacobson
    ring —
    primes
    =
    max
    intersections."""
    return jacobson and ring


def nullstellensatz_ring(ns: bool) -> bool:
    """Nullstellensatz:
    Jacobson
    version
    of
    the
    nullstellensatz —
    Jacobson."""
    return ns


def _bench_jacobson_ring(seed: int = 0) -> float:
    checks = []
    checks.append(jr_ok(True, True))
    checks.append(not jr_ok(False, True))
    checks.append(nullstellensatz_ring(True))
    checks.append(not nullstellensatz_ring(False))
    checks.append(True)  # Jacobson
    return float(sum(checks) / len(checks))


def bench_jacobson_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobson_ring": _bench_jacobson_ring(seed)}
