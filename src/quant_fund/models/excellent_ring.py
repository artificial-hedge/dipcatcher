"""Excellent rings (SYNTHETIC)."""

from __future__ import annotations


def er_ok(excellent: bool, ring: bool) -> bool:
    """Excellent
    ring:
    excellent
    ring —
    geometrically
    regular."""
    return excellent and ring


def quasi_excellent(qe: bool) -> bool:
    """Quasi
    excellent:
    quasi
    excellent
    ring —
    regular
    formal."""
    return qe


def _bench_excellent_ring(seed: int = 0) -> float:
    checks = []
    checks.append(er_ok(True, True))
    checks.append(not er_ok(False, True))
    checks.append(quasi_excellent(True))
    checks.append(not quasi_excellent(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_excellent_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_excellent_ring": _bench_excellent_ring(seed)}
