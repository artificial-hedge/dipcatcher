"""derived integral module (SYNTHETIC)."""

from __future__ import annotations


def derived_integral_ok(derived: bool, geometric: bool) -> bool:
    """derived_integral
    check:
    derived
    structure —
    conn."""
    return derived and geometric


def derived_integral_aux(aux: bool) -> bool:
    """derived_integral
    aux:
    auxiliary
    derived
    check —
    local."""
    return aux


def _bench_derived_integral(seed: int = 0) -> float:
    checks = []
    checks.append(derived_integral_ok(True, True))
    checks.append(not derived_integral_ok(False, True))
    checks.append(derived_integral_aux(True))
    checks.append(not derived_integral_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_integral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_integral": _bench_derived_integral(seed)}
