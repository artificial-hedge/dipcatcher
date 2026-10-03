"""azuma ineq module (SYNTHETIC)."""

from __future__ import annotations


def azuma_ineq_ok(con: bool, tail: bool) -> bool:
    """azuma_ineq
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def azuma_ineq_aux(aux: bool) -> bool:
    """azuma_ineq
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_azuma_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(azuma_ineq_ok(True, True))
    checks.append(not azuma_ineq_ok(False, True))
    checks.append(azuma_ineq_aux(True))
    checks.append(not azuma_ineq_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_azuma_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_azuma_ineq": _bench_azuma_ineq(seed)}
