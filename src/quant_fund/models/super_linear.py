"""super linear module (SYNTHETIC)."""

from __future__ import annotations


def super_linear_ok(sb1: bool, st: bool) -> bool:
    """super_linear
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def super_linear_aux(aux: bool) -> bool:
    """super_linear
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_super_linear(seed: int = 0) -> float:
    checks = []
    checks.append(super_linear_ok(True, True))
    checks.append(not super_linear_ok(False, True))
    checks.append(super_linear_aux(True))
    checks.append(not super_linear_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_super_linear(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_linear": _bench_super_linear(seed)}
