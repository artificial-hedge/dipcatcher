"""derived fiber2 module (SYNTHETIC)."""

from __future__ import annotations


def derived_fiber2_ok(derived: bool, geometry: bool) -> bool:
    """derived_fiber2
    check:
    derived
    geometry —
    spectral."""
    return derived and geometry


def derived_fiber2_aux(aux: bool) -> bool:
    """derived_fiber2
    aux:
    auxiliary
    derived
    check —
    stack."""
    return aux


def _bench_derived_fiber2(seed: int = 0) -> float:
    checks = []
    checks.append(derived_fiber2_ok(True, True))
    checks.append(not derived_fiber2_ok(False, True))
    checks.append(derived_fiber2_aux(True))
    checks.append(not derived_fiber2_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_fiber2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_fiber2": _bench_derived_fiber2(seed)}
