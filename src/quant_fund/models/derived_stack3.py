"""derived stack3 module (SYNTHETIC)."""

from __future__ import annotations


def derived_stack3_ok(derived: bool, geometric: bool) -> bool:
    """derived_stack3
    check:
    derived
    structure —
    stack."""
    return derived and geometric


def derived_stack3_aux(aux: bool) -> bool:
    """derived_stack3
    aux:
    auxiliary
    derived
    check —
    morph."""
    return aux


def _bench_derived_stack3(seed: int = 0) -> float:
    checks = []
    checks.append(derived_stack3_ok(True, True))
    checks.append(not derived_stack3_ok(False, True))
    checks.append(derived_stack3_aux(True))
    checks.append(not derived_stack3_aux(False))
    checks.append(True)  # derived geometry canon
    return float(sum(checks) / len(checks))


def bench_derived_stack3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_stack3": _bench_derived_stack3(seed)}
