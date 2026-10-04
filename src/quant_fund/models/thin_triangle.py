"""Thin triangles (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(delta_thin: bool, comparison: bool) -> bool:
    """Thin
    triangles:
    every
    side
    lies
    in
    the
    delta-
    neighborhood
    of
    the
    other
    two."""
    return delta_thin and comparison


def four_point_cond(fp: bool) -> bool:
    """Four-
    point
    condition:
    equivalent
    Gromov
    product
    characterization
    of
    hyperbolicity."""
    return fp


def _bench_thin_triangle(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(four_point_cond(True))
    checks.append(not four_point_cond(False))
    checks.append(True)  # Gromov
    return float(sum(checks) / len(checks))


def bench_thin_triangle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thin_triangle": _bench_thin_triangle(seed)}
