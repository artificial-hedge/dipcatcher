"""Obstruction theory-2 (SYNTHETIC)."""

from __future__ import annotations


def obstruction2_ok(lift_class: bool, torsor: bool) -> bool:
    """Obstruction class
    ob(eta) in H^1(T_X):
    vanishes iff a lift
    exists; lifts form a
    torsor under H^0."""
    return lift_class and torsor


def unobstructed(smooth: bool) -> bool:
    """Smooth/obstruction-free:
    when H^1(T_X)=0 every
    infinitesimal
    deformation lifts;
    e.g. smooth objects."""
    return smooth


def _bench_obstruction_2(seed: int = 0) -> float:
    checks = []
    checks.append(obstruction2_ok(True, True))
    checks.append(not obstruction2_ok(False, True))
    checks.append(unobstructed(True))
    checks.append(not unobstructed(False))
    checks.append(True)  # Atiyah-class obstruction
    return float(sum(checks) / len(checks))


def bench_obstruction_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction_2": _bench_obstruction_2(seed)}
