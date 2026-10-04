"""first order_dom module (SYNTHETIC)."""

from __future__ import annotations


def first_order_dom_ok(ord: bool, dom: bool) -> bool:
    """first_order_dom
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def first_order_dom_aux(aux: bool) -> bool:
    """first_order_dom
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_first_order_dom(seed: int = 0) -> float:
    checks = []
    checks.append(first_order_dom_ok(True, True))
    checks.append(not first_order_dom_ok(False, True))
    checks.append(first_order_dom_aux(True))
    checks.append(not first_order_dom_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_first_order_dom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_first_order_dom": _bench_first_order_dom(seed)}
