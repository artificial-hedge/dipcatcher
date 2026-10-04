"""lebesgue const module (SYNTHETIC)."""

from __future__ import annotations


def lebesgue_const_ok(step: bool, radius: bool) -> bool:
    """lebesgue_const
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def lebesgue_const_aux(aux: bool) -> bool:
    """lebesgue_const
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_lebesgue_const(seed: int = 0) -> float:
    checks = []
    checks.append(lebesgue_const_ok(True, True))
    checks.append(not lebesgue_const_ok(False, True))
    checks.append(lebesgue_const_aux(True))
    checks.append(not lebesgue_const_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_lebesgue_const(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lebesgue_const": _bench_lebesgue_const(seed)}
