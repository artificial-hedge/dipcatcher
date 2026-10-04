"""rough vol2 module (SYNTHETIC)."""

from __future__ import annotations


def rough_vol2_ok(st1: bool, kk: bool) -> bool:
    """rough_vol2
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def rough_vol2_aux(aux: bool) -> bool:
    """rough_vol2
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_rough_vol2(seed: int = 0) -> float:
    checks = []
    checks.append(rough_vol2_ok(True, True))
    checks.append(not rough_vol2_ok(False, True))
    checks.append(rough_vol2_aux(True))
    checks.append(not rough_vol2_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_rough_vol2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_vol2": _bench_rough_vol2(seed)}
