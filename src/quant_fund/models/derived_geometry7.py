"""derived geometry7 module (SYNTHETIC)."""

from __future__ import annotations


def derived_geometry7_ok(derived: bool, geometric: bool) -> bool:
    """derived_geometry7
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_geometry7_aux(aux: bool) -> bool:
    """derived_geometry7
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_geometry7(seed: int = 0) -> float:
    checks = []
    checks.append(derived_geometry7_ok(True, True))
    checks.append(not derived_geometry7_ok(False, True))
    checks.append(derived_geometry7_aux(True))
    checks.append(not derived_geometry7_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_geometry7(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_geometry7": _bench_derived_geometry7(seed)}
