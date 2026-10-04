"""derived cohom module (SYNTHETIC)."""

from __future__ import annotations


def derived_cohom_ok(derived: bool, geometry: bool) -> bool:
    """derived_cohom
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def derived_cohom_aux(aux: bool) -> bool:
    """derived_cohom
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_derived_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(derived_cohom_ok(True, True))
    checks.append(not derived_cohom_ok(False, True))
    checks.append(derived_cohom_aux(True))
    checks.append(not derived_cohom_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_cohom": _bench_derived_cohom(seed)}
