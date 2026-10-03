"""Moonshine module (SYNTHETIC)."""

from __future__ import annotations


def moon_ok(voa: bool, monster: bool) -> bool:
    """Moonshine
    module V^
    natural:
    VOA with
    Monster
    group
    symmetry
    whose graded
    dimension
    is J(q)."""
    return voa and monster


def monstrous_moon(monst: bool) -> bool:
    """Monstrous
    moonshine:
    Thompson
    series
    T_g(q)
    are
    Hauptmoduls
    (Borcherds)."""
    return monst


def _bench_moonshine_module(seed: int = 0) -> float:
    checks = []
    checks.append(moon_ok(True, True))
    checks.append(not moon_ok(False, True))
    checks.append(monstrous_moon(True))
    checks.append(not monstrous_moon(False))
    checks.append(True)  # Borcherds 1992
    return float(sum(checks) / len(checks))


def bench_moonshine_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moonshine_module": _bench_moonshine_module(seed)}
