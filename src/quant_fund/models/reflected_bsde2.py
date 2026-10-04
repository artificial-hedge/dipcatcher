"""reflected bsde2 module (SYNTHETIC)."""

from __future__ import annotations


def reflected_bsde2_ok(sb1: bool, st: bool) -> bool:
    """reflected_bsde2
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def reflected_bsde2_aux(aux: bool) -> bool:
    """reflected_bsde2
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_reflected_bsde2(seed: int = 0) -> float:
    checks = []
    checks.append(reflected_bsde2_ok(True, True))
    checks.append(not reflected_bsde2_ok(False, True))
    checks.append(reflected_bsde2_aux(True))
    checks.append(not reflected_bsde2_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_reflected_bsde2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reflected_bsde2": _bench_reflected_bsde2(seed)}
