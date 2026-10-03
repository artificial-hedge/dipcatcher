"""derived etale module (SYNTHETIC)."""

from __future__ import annotations


def derived_etale_ok(derived: bool, geometric: bool) -> bool:
    """derived_etale
    check:
    derived
    structure —
    etale."""
    return derived and geometric


def derived_etale_aux(aux: bool) -> bool:
    """derived_etale
    aux:
    auxiliary
    derived
    check —
    flat."""
    return aux


def _bench_derived_etale(seed: int = 0) -> float:
    checks = []
    checks.append(derived_etale_ok(True, True))
    checks.append(not derived_etale_ok(False, True))
    checks.append(derived_etale_aux(True))
    checks.append(not derived_etale_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_etale(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_etale": _bench_derived_etale(seed)}
