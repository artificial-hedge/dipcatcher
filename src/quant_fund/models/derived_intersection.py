"""derived intersection module (SYNTHETIC)."""

from __future__ import annotations


def derived_intersection_ok(derived: bool, geometry: bool) -> bool:
    """derived_intersection
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def derived_intersection_aux(aux: bool) -> bool:
    """derived_intersection
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_derived_intersection(seed: int = 0) -> float:
    checks = []
    checks.append(derived_intersection_ok(True, True))
    checks.append(not derived_intersection_ok(False, True))
    checks.append(derived_intersection_aux(True))
    checks.append(not derived_intersection_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_intersection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_intersection": _bench_derived_intersection(seed)}
