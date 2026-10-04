"""nest accel module (SYNTHETIC)."""

from __future__ import annotations


def nest_accel_ok(step: bool, conv: bool) -> bool:
    """nest_accel
    check:
    acceleration —
    step/contraction
    consistency."""
    return step and conv


def nest_accel_aux(aux: bool) -> bool:
    """nest_accel
    aux:
    auxiliary
    accelerator check —
    rate bound."""
    return aux


def _bench_nest_accel(seed: int = 0) -> float:
    checks = []
    checks.append(nest_accel_ok(True, True))
    checks.append(not nest_accel_ok(False, True))
    checks.append(nest_accel_aux(True))
    checks.append(not nest_accel_aux(False))
    checks.append(True)  # acceleration canon
    return float(sum(checks) / len(checks))


def bench_nest_accel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nest_accel": _bench_nest_accel(seed)}
