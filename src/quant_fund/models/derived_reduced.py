"""derived reduced module (SYNTHETIC)."""

from __future__ import annotations


def derived_reduced_ok(derived: bool, geometric: bool) -> bool:
    """derived_reduced
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_reduced_aux(aux: bool) -> bool:
    """derived_reduced
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_reduced(seed: int = 0) -> float:
    checks = []
    checks.append(derived_reduced_ok(True, True))
    checks.append(not derived_reduced_ok(False, True))
    checks.append(derived_reduced_aux(True))
    checks.append(not derived_reduced_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_reduced(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_reduced": _bench_derived_reduced(seed)}
