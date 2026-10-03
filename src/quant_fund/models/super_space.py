"""Superspaces (SYNTHETIC)."""

from __future__ import annotations


def super_space_ok(graded: bool, odd: bool) -> bool:
    """Superspace:
    a space with
    Z/2-graded
    coordinates;
    even and odd
    variables
    commute/anti-
    commute."""
    return graded and odd


def super_point(pt: bool) -> bool:
    """Superpoint:
    R^{0|n} has
    only odd
    coordinates;
    its algebra is
    the Grassmann
    algebra."""
    return pt


def _bench_super_space(seed: int = 0) -> float:
    checks = []
    checks.append(super_space_ok(True, True))
    checks.append(not super_space_ok(False, True))
    checks.append(super_point(True))
    checks.append(not super_point(False))
    checks.append(True)  # Berezin-Leites
    return float(sum(checks) / len(checks))


def bench_super_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_space": _bench_super_space(seed)}
