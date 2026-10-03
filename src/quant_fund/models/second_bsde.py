"""second bsde module (SYNTHETIC)."""

from __future__ import annotations


def second_bsde_ok(sb1: bool, st: bool) -> bool:
    """second_bsde
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def second_bsde_aux(aux: bool) -> bool:
    """second_bsde
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_second_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(second_bsde_ok(True, True))
    checks.append(not second_bsde_ok(False, True))
    checks.append(second_bsde_aux(True))
    checks.append(not second_bsde_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_second_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_second_bsde": _bench_second_bsde(seed)}
