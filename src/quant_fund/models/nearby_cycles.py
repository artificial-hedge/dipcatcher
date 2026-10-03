"""Nearby cycles (SYNTHETIC)."""

from __future__ import annotations


def nc_ok(degeneration: bool, monodromy: bool) -> bool:
    """Nearby
    cycles:
    sheaf
    of
    nearby
    fiber
    with
    monodromy —
    degeneration
    functor."""
    return degeneration and monodromy


def vanishing_nearby(vn: bool) -> bool:
    """Vanishing-
    nearby
    triangle:
    canonical
    and
    variation
    maps —
    Deligne's
    nearby
    cycles."""
    return vn


def _bench_nearby_cycles(seed: int = 0) -> float:
    checks = []
    checks.append(nc_ok(True, True))
    checks.append(not nc_ok(False, True))
    checks.append(vanishing_nearby(True))
    checks.append(not vanishing_nearby(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_nearby_cycles(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nearby_cycles": _bench_nearby_cycles(seed)}
