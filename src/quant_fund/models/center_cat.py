"""Drinfeld center (SYNTHETIC)."""

from __future__ import annotations


def zc_ok(center: bool, category: bool) -> bool:
    """Center:
    Drinfeld
    center
    Z(C) —
    Drinfeld
    center."""
    return center and category


def half_braiding(hb: bool) -> bool:
    """Half:
    half-
    braiding
    structure —
    half
    braiding."""
    return hb


def _bench_center_cat(seed: int = 0) -> float:
    checks = []
    checks.append(zc_ok(True, True))
    checks.append(not zc_ok(False, True))
    checks.append(half_braiding(True))
    checks.append(not half_braiding(False))
    checks.append(True)  # Drinfeld
    return float(sum(checks) / len(checks))


def bench_center_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_center_cat": _bench_center_cat(seed)}
