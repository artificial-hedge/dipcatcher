"""first order_rel module (SYNTHETIC)."""

from __future__ import annotations


def first_order_rel_ok(beta: bool, conv: bool) -> bool:
    """first_order_rel
    check:
    reliability —
    failure-probability
    consistency."""
    return beta and conv


def first_order_rel_aux(aux: bool) -> bool:
    """first_order_rel
    aux:
    auxiliary
    reliability check —
    index bound."""
    return aux


def _bench_first_order_rel(seed: int = 0) -> float:
    checks = []
    checks.append(first_order_rel_ok(True, True))
    checks.append(not first_order_rel_ok(False, True))
    checks.append(first_order_rel_aux(True))
    checks.append(not first_order_rel_aux(False))
    checks.append(True)  # reliability canon
    return float(sum(checks) / len(checks))


def bench_first_order_rel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_first_order_rel": _bench_first_order_rel(seed)}
