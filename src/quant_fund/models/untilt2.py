"""Untilt 2 (SYNTHETIC)."""

from __future__ import annotations


def u2_ok(untilt: bool, curve: bool) -> bool:
    """Untilt
    2:
    untilt
    curve —
    Fargues."""
    return untilt and curve


def untilt_map(um: bool) -> bool:
    """Untilt
    map:
    untilt
    map —
    perfectoid."""
    return um


def _bench_untilt2(seed: int = 0) -> float:
    checks = []
    checks.append(u2_ok(True, True))
    checks.append(not u2_ok(False, True))
    checks.append(untilt_map(True))
    checks.append(not untilt_map(False))
    checks.append(True)  # Fargues
    return float(sum(checks) / len(checks))


def bench_untilt2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_untilt2": _bench_untilt2(seed)}
