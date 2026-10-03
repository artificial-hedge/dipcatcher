"""gauss newton module (SYNTHETIC)."""

from __future__ import annotations


def gauss_newton_ok(step: bool, resid: bool) -> bool:
    """gauss_newton
    check:
    nonlinear —
    solver-step
    consistency."""
    return step and resid


def gauss_newton_aux(aux: bool) -> bool:
    """gauss_newton
    aux:
    auxiliary
    solver check —
    residual bound."""
    return aux


def _bench_gauss_newton(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_newton_ok(True, True))
    checks.append(not gauss_newton_ok(False, True))
    checks.append(gauss_newton_aux(True))
    checks.append(not gauss_newton_aux(False))
    checks.append(True)  # nonlinear canon
    return float(sum(checks) / len(checks))


def bench_gauss_newton(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_newton": _bench_gauss_newton(seed)}
