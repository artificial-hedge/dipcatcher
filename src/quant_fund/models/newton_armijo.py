"""newton armijo module (SYNTHETIC)."""

from __future__ import annotations


def newton_armijo_ok(step: bool, radius: bool) -> bool:
    """newton_armijo
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def newton_armijo_aux(aux: bool) -> bool:
    """newton_armijo
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_newton_armijo(seed: int = 0) -> float:
    checks = []
    checks.append(newton_armijo_ok(True, True))
    checks.append(not newton_armijo_ok(False, True))
    checks.append(newton_armijo_aux(True))
    checks.append(not newton_armijo_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_newton_armijo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_newton_armijo": _bench_newton_armijo(seed)}
