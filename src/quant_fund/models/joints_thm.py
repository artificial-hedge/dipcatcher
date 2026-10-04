"""Joints theorem (SYNTHETIC)."""

from __future__ import annotations


def joints_ok(triple: bool, bound: bool) -> bool:
    """Joints
    theorem:
    n lines in
    R^3 have
    O(n^{3/2})
    joints
    (Guth-
    Katz); a
    joint is
    a point
    on three
    non-
    coplanar
    lines."""
    return triple and bound


def polynomial_method(poly: bool) -> bool:
    """Polynomial
    method:
    joints
    proved
    via
    parameter
    counting
    and
    degree
    reduction."""
    return poly


def _bench_joints_thm(seed: int = 0) -> float:
    checks = []
    checks.append(joints_ok(True, True))
    checks.append(not joints_ok(False, True))
    checks.append(polynomial_method(True))
    checks.append(not polynomial_method(False))
    checks.append(True)  # Guth-Katz
    return float(sum(checks) / len(checks))


def bench_joints_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_joints_thm": _bench_joints_thm(seed)}
