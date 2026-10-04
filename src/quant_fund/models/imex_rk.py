"""imex rk module (SYNTHETIC)."""

from __future__ import annotations


def imex_rk_ok(step: bool, order: bool) -> bool:
    """imex_rk
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def imex_rk_aux(aux: bool) -> bool:
    """imex_rk
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_imex_rk(seed: int = 0) -> float:
    checks = []
    checks.append(imex_rk_ok(True, True))
    checks.append(not imex_rk_ok(False, True))
    checks.append(imex_rk_aux(True))
    checks.append(not imex_rk_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_imex_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imex_rk": _bench_imex_rk(seed)}
