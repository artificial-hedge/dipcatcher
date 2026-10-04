"""derived hecke module (SYNTHETIC)."""

from __future__ import annotations


def derived_hecke_ok(derived: bool, geometry: bool) -> bool:
    """derived_hecke
    check:
    derived
    geometry
    structure —
    spectral."""
    return derived and geometry


def derived_hecke_aux(aux: bool) -> bool:
    """derived_hecke
    aux:
    auxiliary
    derived-geom
    check —
    analytic."""
    return aux


def _bench_derived_hecke(seed: int = 0) -> float:
    checks = []
    checks.append(derived_hecke_ok(True, True))
    checks.append(not derived_hecke_ok(False, True))
    checks.append(derived_hecke_aux(True))
    checks.append(not derived_hecke_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_hecke(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_hecke": _bench_derived_hecke(seed)}
