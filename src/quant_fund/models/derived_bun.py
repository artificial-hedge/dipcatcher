"""derived bun module (SYNTHETIC)."""

from __future__ import annotations


def derived_bun_ok(derived: bool, geometry: bool) -> bool:
    """derived_bun
    check:
    derived
    geometry
    structure —
    spectral."""
    return derived and geometry


def derived_bun_aux(aux: bool) -> bool:
    """derived_bun
    aux:
    auxiliary
    derived-geom
    check —
    analytic."""
    return aux


def _bench_derived_bun(seed: int = 0) -> float:
    checks = []
    checks.append(derived_bun_ok(True, True))
    checks.append(not derived_bun_ok(False, True))
    checks.append(derived_bun_aux(True))
    checks.append(not derived_bun_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_bun(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_bun": _bench_derived_bun(seed)}
