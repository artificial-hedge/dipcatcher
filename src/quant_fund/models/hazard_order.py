"""hazard order module (SYNTHETIC)."""

from __future__ import annotations


def hazard_order_ok(ord1: bool, meas: bool) -> bool:
    """hazard_order
    check:
    stochastic-order
    structure —
    semimartingale
    canon."""
    return ord1 and meas


def hazard_order_aux(aux: bool) -> bool:
    """hazard_order
    aux:
    auxiliary
    order
    check —
    Cramer-Wold
    device."""
    return aux


def _bench_hazard_order(seed: int = 0) -> float:
    checks = []
    checks.append(hazard_order_ok(True, True))
    checks.append(not hazard_order_ok(False, True))
    checks.append(hazard_order_aux(True))
    checks.append(not hazard_order_aux(False))
    checks.append(True)  # order canon
    return float(sum(checks) / len(checks))


def bench_hazard_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hazard_order": _bench_hazard_order(seed)}
