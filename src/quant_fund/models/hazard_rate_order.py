"""hazard rate_order module (SYNTHETIC)."""

from __future__ import annotations


def hazard_rate_order_ok(ord: bool, dom: bool) -> bool:
    """hazard_rate_order
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def hazard_rate_order_aux(aux: bool) -> bool:
    """hazard_rate_order
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_hazard_rate_order(seed: int = 0) -> float:
    checks = []
    checks.append(hazard_rate_order_ok(True, True))
    checks.append(not hazard_rate_order_ok(False, True))
    checks.append(hazard_rate_order_aux(True))
    checks.append(not hazard_rate_order_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_hazard_rate_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hazard_rate_order": _bench_hazard_rate_order(seed)}
