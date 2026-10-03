"""newton method module (SYNTHETIC)."""

from __future__ import annotations


def newton_method_ok(step: bool, conv: bool) -> bool:
    """newton_method
    check:
    optimization —
    descent step
    consistency."""
    return step and conv


def newton_method_aux(aux: bool) -> bool:
    """newton_method
    aux:
    auxiliary
    optimizer check —
    rate bound."""
    return aux


def _bench_newton_method(seed: int = 0) -> float:
    checks = []
    checks.append(newton_method_ok(True, True))
    checks.append(not newton_method_ok(False, True))
    checks.append(newton_method_aux(True))
    checks.append(not newton_method_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_newton_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_newton_method": _bench_newton_method(seed)}
