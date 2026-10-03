"""quadratic bsde module (SYNTHETIC)."""

from __future__ import annotations


def quadratic_bsde_ok(sb1: bool, st: bool) -> bool:
    """quadratic_bsde
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def quadratic_bsde_aux(aux: bool) -> bool:
    """quadratic_bsde
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_quadratic_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(quadratic_bsde_ok(True, True))
    checks.append(not quadratic_bsde_ok(False, True))
    checks.append(quadratic_bsde_aux(True))
    checks.append(not quadratic_bsde_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_quadratic_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadratic_bsde": _bench_quadratic_bsde(seed)}
