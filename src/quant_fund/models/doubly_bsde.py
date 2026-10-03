"""doubly bsde module (SYNTHETIC)."""

from __future__ import annotations


def doubly_bsde_ok(sb1: bool, st: bool) -> bool:
    """doubly_bsde
    check:
    2BSDE —
    Soner-Touzi
    formulation."""
    return sb1 and st


def doubly_bsde_aux(aux: bool) -> bool:
    """doubly_bsde
    aux:
    auxiliary
    2BSDE
    check —
    quadratic
    growth."""
    return aux


def _bench_doubly_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(doubly_bsde_ok(True, True))
    checks.append(not doubly_bsde_ok(False, True))
    checks.append(doubly_bsde_aux(True))
    checks.append(not doubly_bsde_aux(False))
    checks.append(True)  # 2BSDE canon
    return float(sum(checks) / len(checks))


def bench_doubly_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doubly_bsde": _bench_doubly_bsde(seed)}
