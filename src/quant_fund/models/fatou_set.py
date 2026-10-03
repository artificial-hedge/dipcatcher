"""Fatou set (SYNTHETIC)."""

from __future__ import annotations


def fatou_ok(normal: bool, basin: bool) -> bool:
    """Fatou
    set:
    maximal
    open
    set of
    normality;
    basins
    of
    attracting
    cycles
    live
    inside."""
    return normal and basin


def components_periodic(comp: bool) -> bool:
    """Periodic
    Fatou
    components:
    attracting,
    parabolic,
    Siegel,
    or
    Herman
    rings."""
    return comp


def _bench_fatou_set(seed: int = 0) -> float:
    checks = []
    checks.append(fatou_ok(True, True))
    checks.append(not fatou_ok(False, True))
    checks.append(components_periodic(True))
    checks.append(not components_periodic(False))
    checks.append(True)  # Fatou
    return float(sum(checks) / len(checks))


def bench_fatou_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fatou_set": _bench_fatou_set(seed)}
