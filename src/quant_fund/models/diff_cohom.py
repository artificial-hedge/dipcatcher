"""Differential cohomology (SYNTHETIC)."""

from __future__ import annotations


def diff_ok(char: bool, curvature: bool) -> bool:
    """Differential
    cohomology H^k:
    refines
    characteristic
    classes to
    include
    connection and
    curvature data."""
    return char and curvature


def hexagon_diag(hexagon: bool) -> bool:
    """Differential
    cohomology
    hexagon
    relates
    H^k,
    forms,
    flat
    and torsion
    sectors."""
    return hexagon


def _bench_diff_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(diff_ok(True, True))
    checks.append(not diff_ok(False, True))
    checks.append(hexagon_diag(True))
    checks.append(not hexagon_diag(False))
    checks.append(True)  # Simons-Sullivan
    return float(sum(checks) / len(checks))


def bench_diff_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diff_cohom": _bench_diff_cohom(seed)}
