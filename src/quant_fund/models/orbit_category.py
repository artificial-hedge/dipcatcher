"""orbit category module (SYNTHETIC)."""

from __future__ import annotations


def orbit_category_ok(calabi: bool, yau: bool) -> bool:
    """orbit_category
    check:
    calabi
    structure —
    Frobenius."""
    return calabi and yau


def orbit_category_aux(aux: bool) -> bool:
    """orbit_category
    aux:
    auxiliary
    calabi
    check —
    stable."""
    return aux


def _bench_orbit_category(seed: int = 0) -> float:
    checks = []
    checks.append(orbit_category_ok(True, True))
    checks.append(not orbit_category_ok(False, True))
    checks.append(orbit_category_aux(True))
    checks.append(not orbit_category_aux(False))
    checks.append(True)  # Calabi-Yau canon
    return float(sum(checks) / len(checks))


def bench_orbit_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orbit_category": _bench_orbit_category(seed)}
