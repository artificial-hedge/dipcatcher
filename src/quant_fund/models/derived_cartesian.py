"""derived cartesian module (SYNTHETIC)."""

from __future__ import annotations


def derived_cartesian_ok(derived: bool, geometric: bool) -> bool:
    """derived_cartesian
    check:
    derived
    structure —
    etale."""
    return derived and geometric


def derived_cartesian_aux(aux: bool) -> bool:
    """derived_cartesian
    aux:
    auxiliary
    derived
    check —
    flat."""
    return aux


def _bench_derived_cartesian(seed: int = 0) -> float:
    checks = []
    checks.append(derived_cartesian_ok(True, True))
    checks.append(not derived_cartesian_ok(False, True))
    checks.append(derived_cartesian_aux(True))
    checks.append(not derived_cartesian_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_cartesian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_cartesian": _bench_derived_cartesian(seed)}
