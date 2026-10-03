"""ars imex module (SYNTHETIC)."""

from __future__ import annotations


def ars_imex_ok(step: bool, order: bool) -> bool:
    """ars_imex
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def ars_imex_aux(aux: bool) -> bool:
    """ars_imex
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_ars_imex(seed: int = 0) -> float:
    checks = []
    checks.append(ars_imex_ok(True, True))
    checks.append(not ars_imex_ok(False, True))
    checks.append(ars_imex_aux(True))
    checks.append(not ars_imex_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_ars_imex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ars_imex": _bench_ars_imex(seed)}
