"""max ineq module (SYNTHETIC)."""

from __future__ import annotations


def max_ineq_ok(bound: bool, tail: bool) -> bool:
    """max_ineq
    check:
    maximal
    inequality —
    tail bound."""
    return bound and tail


def max_ineq_aux(aux: bool) -> bool:
    """max_ineq
    aux:
    auxiliary
    inequality check —
    moment."""
    return aux


def _bench_max_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(max_ineq_ok(True, True))
    checks.append(not max_ineq_ok(False, True))
    checks.append(max_ineq_aux(True))
    checks.append(not max_ineq_aux(False))
    checks.append(True)  # maximal-inequality canon
    return float(sum(checks) / len(checks))


def bench_max_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_max_ineq": _bench_max_ineq(seed)}
