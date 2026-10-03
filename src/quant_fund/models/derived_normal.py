"""derived normal module (SYNTHETIC)."""

from __future__ import annotations


def derived_normal_ok(derived: bool, geometric: bool) -> bool:
    """derived_normal
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_normal_aux(aux: bool) -> bool:
    """derived_normal
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_normal(seed: int = 0) -> float:
    checks = []
    checks.append(derived_normal_ok(True, True))
    checks.append(not derived_normal_ok(False, True))
    checks.append(derived_normal_aux(True))
    checks.append(not derived_normal_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_normal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_normal": _bench_derived_normal(seed)}
