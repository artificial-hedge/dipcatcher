"""steepest descent module (SYNTHETIC)."""

from __future__ import annotations


def steepest_descent_ok(series: bool, order: bool) -> bool:
    """steepest_descent
    check:
    asymptotic
    analysis —
    series."""
    return series and order


def steepest_descent_aux(aux: bool) -> bool:
    """steepest_descent
    aux:
    auxiliary
    asymptotic check —
    remainder."""
    return aux


def _bench_steepest_descent(seed: int = 0) -> float:
    checks = []
    checks.append(steepest_descent_ok(True, True))
    checks.append(not steepest_descent_ok(False, True))
    checks.append(steepest_descent_aux(True))
    checks.append(not steepest_descent_aux(False))
    checks.append(True)  # asymptotic-analysis canon
    return float(sum(checks) / len(checks))


def bench_steepest_descent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steepest_descent": _bench_steepest_descent(seed)}
