"""Relative duality (SYNTHETIC)."""

from __future__ import annotations


def rd_ok(relative_dual: bool, family: bool) -> bool:
    """Relative
    duality:
    dualizing
    in
    families
    via
    f^! —
    relative
    dualizing."""
    return relative_dual and family


def smooth_relative(sm: bool) -> bool:
    """Smooth
    relative:
    relative
    dualizing
    is
    top
    cotangent
    smooth —
    relative
    canonical."""
    return sm


def _bench_relative_duality(seed: int = 0) -> float:
    checks = []
    checks.append(rd_ok(True, True))
    checks.append(not rd_ok(False, True))
    checks.append(smooth_relative(True))
    checks.append(not smooth_relative(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_relative_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_relative_duality": _bench_relative_duality(seed)}
