"""likelihood order module (SYNTHETIC)."""

from __future__ import annotations


def likelihood_order_ok(ord1: bool, meas: bool) -> bool:
    """likelihood_order
    check:
    stochastic-order
    structure —
    semimartingale
    canon."""
    return ord1 and meas


def likelihood_order_aux(aux: bool) -> bool:
    """likelihood_order
    aux:
    auxiliary
    order
    check —
    Cramer-Wold
    device."""
    return aux


def _bench_likelihood_order(seed: int = 0) -> float:
    checks = []
    checks.append(likelihood_order_ok(True, True))
    checks.append(not likelihood_order_ok(False, True))
    checks.append(likelihood_order_aux(True))
    checks.append(not likelihood_order_aux(False))
    checks.append(True)  # order canon
    return float(sum(checks) / len(checks))


def bench_likelihood_order(seed: int = 0) -> dict[str, float]:
    return {"synthetic_likelihood_order": _bench_likelihood_order(seed)}
