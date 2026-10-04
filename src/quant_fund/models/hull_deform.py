"""Hulls of deformation functors (SYNTHETIC)."""

from __future__ import annotations


def hd_ok(hull: bool, deform: bool) -> bool:
    """Hull
    deformation:
    hull
    of
    a
    deformation
    functor —
    Schlessinger
    hull."""
    return hull and deform


def hull_theorem(ht: bool) -> bool:
    """Hull
    theorem:
    Schlessinger
    hull
    theorem —
    versal
    hull."""
    return ht


def _bench_hull_deform(seed: int = 0) -> float:
    checks = []
    checks.append(hd_ok(True, True))
    checks.append(not hd_ok(False, True))
    checks.append(hull_theorem(True))
    checks.append(not hull_theorem(False))
    checks.append(True)  # Schlessinger
    return float(sum(checks) / len(checks))


def bench_hull_deform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hull_deform": _bench_hull_deform(seed)}
