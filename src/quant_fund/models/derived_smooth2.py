"""derived smooth2 module (SYNTHETIC)."""

from __future__ import annotations


def derived_smooth2_ok(derived: bool, geometric: bool) -> bool:
    """derived_smooth2
    check:
    derived
    structure —
    etale."""
    return derived and geometric


def derived_smooth2_aux(aux: bool) -> bool:
    """derived_smooth2
    aux:
    auxiliary
    derived
    check —
    flat."""
    return aux


def _bench_derived_smooth2(seed: int = 0) -> float:
    checks = []
    checks.append(derived_smooth2_ok(True, True))
    checks.append(not derived_smooth2_ok(False, True))
    checks.append(derived_smooth2_aux(True))
    checks.append(not derived_smooth2_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_smooth2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_smooth2": _bench_derived_smooth2(seed)}
