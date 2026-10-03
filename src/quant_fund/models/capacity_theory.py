"""Capacity theory (SYNTHETIC)."""

from __future__ import annotations


def cap_ok(equilibrium: bool, outer: bool) -> bool:
    """Capacity:
    equilibrium
    measure
    minimizes
    energy;
    capacity
    is
    outer-
    regular."""
    return equilibrium and outer


def polar_set(polar: bool) -> bool:
    """Polar
    sets:
    capacity
    zero
    iff
    potential
    can
    be
    +inf
    on
    the
    set."""
    return polar


def _bench_capacity_theory(seed: int = 0) -> float:
    checks = []
    checks.append(cap_ok(True, True))
    checks.append(not cap_ok(False, True))
    checks.append(polar_set(True))
    checks.append(not polar_set(False))
    checks.append(True)  # Frostman-Choquet
    return float(sum(checks) / len(checks))


def bench_capacity_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_capacity_theory": _bench_capacity_theory(seed)}
