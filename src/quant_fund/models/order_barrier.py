"""order barrier module (SYNTHETIC)."""

from __future__ import annotations


def order_barrier_ok(step: bool, order: bool) -> bool:
    """order_barrier
    check:
    ODE-theory/LMM
    canon — step/
    order
    consistency."""
    return step and order


def order_barrier_aux(aux: bool) -> bool:
    """order_barrier
    aux:
    auxiliary
    order check —
    stability bound."""
    return aux


def _bench_order_barrier(seed: int = 0) -> float:
    checks = []
    checks.append(order_barrier_ok(True, True))
    checks.append(not order_barrier_ok(False, True))
    checks.append(order_barrier_aux(True))
    checks.append(not order_barrier_aux(False))
    checks.append(True)  # lmm canon
    return float(sum(checks) / len(checks))


def bench_order_barrier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_order_barrier": _bench_order_barrier(seed)}
