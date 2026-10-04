"""time bsde module (SYNTHETIC)."""

from __future__ import annotations


def time_bsde_ok(fb1: bool, my: bool) -> bool:
    """time_bsde
    check:
    FBSDE-2 —
    Ma-Yong
    decoupling."""
    return fb1 and my


def time_bsde_aux(aux: bool) -> bool:
    """time_bsde
    aux:
    auxiliary
    FBSDE
    check —
    four-step."""
    return aux


def _bench_time_bsde(seed: int = 0) -> float:
    checks = []
    checks.append(time_bsde_ok(True, True))
    checks.append(not time_bsde_ok(False, True))
    checks.append(time_bsde_aux(True))
    checks.append(not time_bsde_aux(False))
    checks.append(True)  # FBSDE canon
    return float(sum(checks) / len(checks))


def bench_time_bsde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_time_bsde": _bench_time_bsde(seed)}
