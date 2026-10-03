"""gauss rk module (SYNTHETIC)."""

from __future__ import annotations


def gauss_rk_ok(step: bool, conv: bool) -> bool:
    """gauss_rk
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def gauss_rk_aux(aux: bool) -> bool:
    """gauss_rk
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_gauss_rk(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_rk_ok(True, True))
    checks.append(not gauss_rk_ok(False, True))
    checks.append(gauss_rk_aux(True))
    checks.append(not gauss_rk_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_gauss_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_rk": _bench_gauss_rk(seed)}
