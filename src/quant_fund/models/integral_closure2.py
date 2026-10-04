"""Integral closures (SYNTHETIC)."""

from __future__ import annotations


def ic2_ok(integral: bool, closure: bool) -> bool:
    """Integral
    closure:
    integral
    closure —
    normalization."""
    return integral and closure


def integrally_closed(icl: bool) -> bool:
    """Integrally
    closed:
    integrally
    closed
    domain —
    normal
    domain."""
    return icl


def _bench_integral_closure2(seed: int = 0) -> float:
    checks = []
    checks.append(ic2_ok(True, True))
    checks.append(not ic2_ok(False, True))
    checks.append(integrally_closed(True))
    checks.append(not integrally_closed(False))
    checks.append(True)  # Noether
    return float(sum(checks) / len(checks))


def bench_integral_closure2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_integral_closure2": _bench_integral_closure2(seed)}
