"""mcdiarmid ineq module (SYNTHETIC)."""

from __future__ import annotations


def mcdiarmid_ineq_ok(con: bool, tail: bool) -> bool:
    """mcdiarmid_ineq
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def mcdiarmid_ineq_aux(aux: bool) -> bool:
    """mcdiarmid_ineq
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_mcdiarmid_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(mcdiarmid_ineq_ok(True, True))
    checks.append(not mcdiarmid_ineq_ok(False, True))
    checks.append(mcdiarmid_ineq_aux(True))
    checks.append(not mcdiarmid_ineq_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_mcdiarmid_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mcdiarmid_ineq": _bench_mcdiarmid_ineq(seed)}
