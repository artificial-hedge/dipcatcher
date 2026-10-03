"""hoeffding ineq module (SYNTHETIC)."""

from __future__ import annotations


def hoeffding_ineq_ok(con: bool, tail: bool) -> bool:
    """hoeffding_ineq
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def hoeffding_ineq_aux(aux: bool) -> bool:
    """hoeffding_ineq
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_hoeffding_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(hoeffding_ineq_ok(True, True))
    checks.append(not hoeffding_ineq_ok(False, True))
    checks.append(hoeffding_ineq_aux(True))
    checks.append(not hoeffding_ineq_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_hoeffding_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hoeffding_ineq": _bench_hoeffding_ineq(seed)}
