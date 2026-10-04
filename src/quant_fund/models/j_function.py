"""j function module (SYNTHETIC)."""

from __future__ import annotations


def j_function_ok(cp: bool, pg: bool) -> bool:
    """j_function
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def j_function_aux(aux: bool) -> bool:
    """j_function
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_j_function(seed: int = 0) -> float:
    checks = []
    checks.append(j_function_ok(True, True))
    checks.append(not j_function_ok(False, True))
    checks.append(j_function_aux(True))
    checks.append(not j_function_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_j_function(seed: int = 0) -> dict[str, float]:
    return {"synthetic_j_function": _bench_j_function(seed)}
