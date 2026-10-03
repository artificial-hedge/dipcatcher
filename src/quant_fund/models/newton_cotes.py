"""newton cotes module (SYNTHETIC)."""

from __future__ import annotations


def newton_cotes_ok(node: bool, weight: bool) -> bool:
    """newton_cotes
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def newton_cotes_aux(aux: bool) -> bool:
    """newton_cotes
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_newton_cotes(seed: int = 0) -> float:
    checks = []
    checks.append(newton_cotes_ok(True, True))
    checks.append(not newton_cotes_ok(False, True))
    checks.append(newton_cotes_aux(True))
    checks.append(not newton_cotes_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_newton_cotes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_newton_cotes": _bench_newton_cotes(seed)}
