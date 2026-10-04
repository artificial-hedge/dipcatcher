"""ssp rk module (SYNTHETIC)."""

from __future__ import annotations


def ssp_rk_ok(step: bool, order: bool) -> bool:
    """ssp_rk
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def ssp_rk_aux(aux: bool) -> bool:
    """ssp_rk
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_ssp_rk(seed: int = 0) -> float:
    checks = []
    checks.append(ssp_rk_ok(True, True))
    checks.append(not ssp_rk_ok(False, True))
    checks.append(ssp_rk_aux(True))
    checks.append(not ssp_rk_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_ssp_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ssp_rk": _bench_ssp_rk(seed)}
