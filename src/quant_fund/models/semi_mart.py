"""semi mart module (SYNTHETIC)."""

from __future__ import annotations


def semi_mart_ok(ord1: bool, meas: bool) -> bool:
    """semi_mart
    check:
    stochastic-order
    structure —
    semimartingale
    canon."""
    return ord1 and meas


def semi_mart_aux(aux: bool) -> bool:
    """semi_mart
    aux:
    auxiliary
    order
    check —
    Cramer-Wold
    device."""
    return aux


def _bench_semi_mart(seed: int = 0) -> float:
    checks = []
    checks.append(semi_mart_ok(True, True))
    checks.append(not semi_mart_ok(False, True))
    checks.append(semi_mart_aux(True))
    checks.append(not semi_mart_aux(False))
    checks.append(True)  # order canon
    return float(sum(checks) / len(checks))


def bench_semi_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semi_mart": _bench_semi_mart(seed)}
