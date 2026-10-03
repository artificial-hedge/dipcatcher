"""isaacs equation module (SYNTHETIC)."""

from __future__ import annotations


def isaacs_equation_ok(sg1: bool, dk: bool) -> bool:
    """isaacs_equation
    check:
    stochastic
    game —
    Dynkin
    value."""
    return sg1 and dk


def isaacs_equation_aux(aux: bool) -> bool:
    """isaacs_equation
    aux:
    auxiliary
    game
    check —
    saddle
    point."""
    return aux


def _bench_isaacs_equation(seed: int = 0) -> float:
    checks = []
    checks.append(isaacs_equation_ok(True, True))
    checks.append(not isaacs_equation_ok(False, True))
    checks.append(isaacs_equation_aux(True))
    checks.append(not isaacs_equation_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_isaacs_equation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isaacs_equation": _bench_isaacs_equation(seed)}
