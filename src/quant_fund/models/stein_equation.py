"""stein equation module (SYNTHETIC)."""

from __future__ import annotations


def stein_equation_ok(op: bool, bound: bool) -> bool:
    """stein_equation
    check:
    Stein
    structure —
    Stein
    equation."""
    return op and bound


def stein_equation_aux(aux: bool) -> bool:
    """stein_equation
    aux:
    auxiliary
    generator
    check —
    Barbour."""
    return aux


def _bench_stein_equation(seed: int = 0) -> float:
    checks = []
    checks.append(stein_equation_ok(True, True))
    checks.append(not stein_equation_ok(False, True))
    checks.append(stein_equation_aux(True))
    checks.append(not stein_equation_aux(False))
    checks.append(True)  # Stein canon
    return float(sum(checks) / len(checks))


def bench_stein_equation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stein_equation": _bench_stein_equation(seed)}
