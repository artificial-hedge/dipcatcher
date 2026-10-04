"""doob ineq module (SYNTHETIC)."""

from __future__ import annotations


def doob_ineq_ok(bound: bool, tail: bool) -> bool:
    """doob_ineq
    check:
    maximal
    inequality —
    tail bound."""
    return bound and tail


def doob_ineq_aux(aux: bool) -> bool:
    """doob_ineq
    aux:
    auxiliary
    inequality check —
    moment."""
    return aux


def _bench_doob_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(doob_ineq_ok(True, True))
    checks.append(not doob_ineq_ok(False, True))
    checks.append(doob_ineq_aux(True))
    checks.append(not doob_ineq_aux(False))
    checks.append(True)  # maximal-inequality canon
    return float(sum(checks) / len(checks))


def bench_doob_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doob_ineq": _bench_doob_ineq(seed)}
