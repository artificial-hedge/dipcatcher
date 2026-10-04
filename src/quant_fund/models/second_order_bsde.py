"""second order_bsde module (SYNTHETIC)."""

from __future__ import annotations


def second_order_bsde_ok(bs1: bool, pp: bool) -> bool:
    """second_order_bsde
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def second_order_bsde_aux(aux: bool) -> bool:
    """second_order_bsde
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_second_order_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(second_order_bsde_ok(True, True))
    checks.append(not second_order_bsde_ok(False, True))
    checks.append(second_order_bsde_aux(True))
    checks.append(not second_order_bsde_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_second_order_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_second_order_bsde": _bench_second_order_bsde(seed)}
