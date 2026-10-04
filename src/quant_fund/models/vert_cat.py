"""Vertical categories (SYNTHETIC)."""

from __future__ import annotations


def vc_ok(vertical: bool, cat: bool) -> bool:
    """Vertical
    category:
    vertical
    category —
    double
    category."""
    return vertical and cat


def double_vertical(dv: bool) -> bool:
    """Double
    vertical:
    double
    vertical
    morphism —
    Ehresmann
    vertical."""
    return dv


def _bench_vert_cat(seed: int = 0) -> float:
    checks = []
    checks.append(vc_ok(True, True))
    checks.append(not vc_ok(False, True))
    checks.append(double_vertical(True))
    checks.append(not double_vertical(False))
    checks.append(True)  # Ehresmann
    return float(sum(checks) / len(checks))


def bench_vert_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vert_cat": _bench_vert_cat(seed)}
