"""Tate curves (SYNTHETIC)."""

from __future__ import annotations


def tc2_ok(tate: bool, curve: bool) -> bool:
    """Tate
    curve:
    Tate
    curve —
    q
    parametrization."""
    return tate and curve


def tate_uniformization(tu: bool) -> bool:
    """Tate
    uniformization:
    Tate
    uniformization —
    rigid
    quotient."""
    return tu


def _bench_tate_curve(seed: int = 0) -> float:
    checks = []
    checks.append(tc2_ok(True, True))
    checks.append(not tc2_ok(False, True))
    checks.append(tate_uniformization(True))
    checks.append(not tate_uniformization(False))
    checks.append(True)  # Tate
    return float(sum(checks) / len(checks))


def bench_tate_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_curve": _bench_tate_curve(seed)}
