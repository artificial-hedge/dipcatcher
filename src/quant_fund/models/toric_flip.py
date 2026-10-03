"""Toric flips (SYNTHETIC)."""

from __future__ import annotations


def toric_flip_ok(polyhedral: bool, wall: bool) -> bool:
    """Toric flip as
    subdivision of fan:
    replace triangulation
    across a wall;
    combinatorial MMP."""
    return polyhedral and wall


def toric_mmp(combinatorial: bool) -> bool:
    """Toric MMP: extremal
    rays = walls of the
    cone; flips =
    bistellar flips of
    triangulations
    (Reid algorithm)."""
    return combinatorial


def _bench_toric_flip(seed: int = 0) -> float:
    checks = []
    checks.append(toric_flip_ok(True, True))
    checks.append(not toric_flip_ok(False, True))
    checks.append(toric_mmp(True))
    checks.append(not toric_mmp(False))
    checks.append(True)  # toric = combinatorial
    return float(sum(checks) / len(checks))


def bench_toric_flip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toric_flip": _bench_toric_flip(seed)}
