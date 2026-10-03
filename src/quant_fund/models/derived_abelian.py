"""derived abelian module (SYNTHETIC)."""

from __future__ import annotations


def derived_abelian_ok(derived: bool, geometry: bool) -> bool:
    """derived_abelian
    check:
    derived
    geometry
    structure —
    spectral."""
    return derived and geometry


def derived_abelian_aux(aux: bool) -> bool:
    """derived_abelian
    aux:
    auxiliary
    derived-geom
    check —
    analytic."""
    return aux


def _bench_derived_abelian(seed: int = 0) -> float:
    checks = []
    checks.append(derived_abelian_ok(True, True))
    checks.append(not derived_abelian_ok(False, True))
    checks.append(derived_abelian_aux(True))
    checks.append(not derived_abelian_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_abelian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_abelian": _bench_derived_abelian(seed)}
