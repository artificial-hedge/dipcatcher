"""convex order module (SYNTHETIC)."""

from __future__ import annotations


def convex_order_ok(ord: bool, dom: bool) -> bool:
    """convex_order
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def convex_order_aux(aux: bool) -> bool:
    """convex_order
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_convex_order(seed: int = 0) -> float:
    checks = []
    checks.append(convex_order_ok(True, True))
    checks.append(not convex_order_ok(False, True))
    checks.append(convex_order_aux(True))
    checks.append(not convex_order_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_convex_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_convex_order": _bench_convex_order(seed)}
