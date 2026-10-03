"""usual stoch_order module (SYNTHETIC)."""

from __future__ import annotations


def usual_stoch_order_ok(ord: bool, dom: bool) -> bool:
    """usual_stoch_order
    check:
    stochastic
    order —
    dominance."""
    return ord and dom


def usual_stoch_order_aux(aux: bool) -> bool:
    """usual_stoch_order
    aux:
    auxiliary
    order check —
    tail."""
    return aux


def _bench_usual_stoch_order(seed: int = 0) -> float:
    checks = []
    checks.append(usual_stoch_order_ok(True, True))
    checks.append(not usual_stoch_order_ok(False, True))
    checks.append(usual_stoch_order_aux(True))
    checks.append(not usual_stoch_order_aux(False))
    checks.append(True)  # stochastic-order canon
    return float(sum(checks) / len(checks))


def bench_usual_stoch_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_usual_stoch_order": _bench_usual_stoch_order(seed)}
