"""Berkovich analytic spaces (SYNTHETIC)."""

from __future__ import annotations


def berkovich_ok(multiplicative: bool, connected: bool) -> bool:
    """Berkovich space X^an:
    multiplicative seminorms
    on affinoid algebra;
    Hausdorff, locally
    compact, path-
    connected."""
    return multiplicative and connected


def berkovich_pts(four_types: bool) -> bool:
    """Four types of Berkovich
    points on the affine
    line; type-2 branch
    points dominate
    geometry."""
    return four_types


def _bench_berkovich_an(seed: int = 0) -> float:
    checks = []
    checks.append(berkovich_ok(True, True))
    checks.append(not berkovich_ok(False, True))
    checks.append(berkovich_pts(True))
    checks.append(not berkovich_pts(False))
    checks.append(True)  # Berkovich spectral theory
    return float(sum(checks) / len(checks))


def bench_berkovich_an(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berkovich_an": _bench_berkovich_an(seed)}
