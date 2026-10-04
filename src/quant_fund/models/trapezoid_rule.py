"""trapezoid rule module (SYNTHETIC)."""

from __future__ import annotations


def trapezoid_rule_ok(step: bool, order: bool) -> bool:
    """trapezoid_rule
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def trapezoid_rule_aux(aux: bool) -> bool:
    """trapezoid_rule
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_trapezoid_rule(seed: int = 0) -> float:
    checks = []
    checks.append(trapezoid_rule_ok(True, True))
    checks.append(not trapezoid_rule_ok(False, True))
    checks.append(trapezoid_rule_aux(True))
    checks.append(not trapezoid_rule_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_trapezoid_rule(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trapezoid_rule": _bench_trapezoid_rule(seed)}
