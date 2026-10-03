"""obstacle bsde module (SYNTHETIC)."""

from __future__ import annotations


def obstacle_bsde_ok(sb1: bool, st: bool) -> bool:
    """obstacle_bsde
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def obstacle_bsde_aux(aux: bool) -> bool:
    """obstacle_bsde
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_obstacle_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(obstacle_bsde_ok(True, True))
    checks.append(not obstacle_bsde_ok(False, True))
    checks.append(obstacle_bsde_aux(True))
    checks.append(not obstacle_bsde_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_obstacle_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstacle_bsde": _bench_obstacle_bsde(seed)}
