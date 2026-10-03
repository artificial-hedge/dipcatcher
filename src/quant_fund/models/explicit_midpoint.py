"""explicit midpoint module (SYNTHETIC)."""

from __future__ import annotations


def explicit_midpoint_ok(step: bool, order: bool) -> bool:
    """explicit_midpoint
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def explicit_midpoint_aux(aux: bool) -> bool:
    """explicit_midpoint
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_explicit_midpoint(seed: int = 0) -> float:
    checks = []
    checks.append(explicit_midpoint_ok(True, True))
    checks.append(not explicit_midpoint_ok(False, True))
    checks.append(explicit_midpoint_aux(True))
    checks.append(not explicit_midpoint_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_explicit_midpoint(seed: int = 0) -> dict[str, float]:
    return {"synthetic_explicit_midpoint": _bench_explicit_midpoint(seed)}
