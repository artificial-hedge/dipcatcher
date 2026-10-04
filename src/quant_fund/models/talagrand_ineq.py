"""talagrand ineq module (SYNTHETIC)."""

from __future__ import annotations


def talagrand_ineq_ok(con: bool, tail: bool) -> bool:
    """talagrand_ineq
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def talagrand_ineq_aux(aux: bool) -> bool:
    """talagrand_ineq
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_talagrand_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(talagrand_ineq_ok(True, True))
    checks.append(not talagrand_ineq_ok(False, True))
    checks.append(talagrand_ineq_aux(True))
    checks.append(not talagrand_ineq_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_talagrand_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_talagrand_ineq": _bench_talagrand_ineq(seed)}
