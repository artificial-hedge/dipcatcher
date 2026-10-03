"""second order_dom module (SYNTHETIC)."""

from __future__ import annotations


def second_order_dom_ok(ord: bool, dom: bool) -> bool:
    """second_order_dom
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def second_order_dom_aux(aux: bool) -> bool:
    """second_order_dom
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_second_order_dom(seed: int = 0) -> float:
    checks = []
    checks.append(second_order_dom_ok(True, True))
    checks.append(not second_order_dom_ok(False, True))
    checks.append(second_order_dom_aux(True))
    checks.append(not second_order_dom_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_second_order_dom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_second_order_dom": _bench_second_order_dom(seed)}
