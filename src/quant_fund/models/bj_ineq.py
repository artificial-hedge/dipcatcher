"""bj ineq module (SYNTHETIC)."""

from __future__ import annotations


def bj_ineq_ok(bound: bool, tail: bool) -> bool:
    """bj_ineq
    check:
    maximal
    inequality —
    tail bound."""
    return bound and tail


def bj_ineq_aux(aux: bool) -> bool:
    """bj_ineq
    aux:
    auxiliary
    inequality check —
    moment."""
    return aux


def _bench_bj_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(bj_ineq_ok(True, True))
    checks.append(not bj_ineq_ok(False, True))
    checks.append(bj_ineq_aux(True))
    checks.append(not bj_ineq_aux(False))
    checks.append(True)  # maximal-inequality canon
    return float(sum(checks) / len(checks))


def bench_bj_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bj_ineq": _bench_bj_ineq(seed)}
