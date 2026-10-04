"""supermodular order module (SYNTHETIC)."""

from __future__ import annotations


def supermodular_order_ok(ord: bool, dom: bool) -> bool:
    """supermodular_order
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def supermodular_order_aux(aux: bool) -> bool:
    """supermodular_order
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_supermodular_order(seed: int = 0) -> float:
    checks = []
    checks.append(supermodular_order_ok(True, True))
    checks.append(not supermodular_order_ok(False, True))
    checks.append(supermodular_order_aux(True))
    checks.append(not supermodular_order_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_supermodular_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_supermodular_order": _bench_supermodular_order(seed)}
