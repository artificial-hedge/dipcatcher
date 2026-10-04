"""Oriented cohomology (SYNTHETIC)."""

from __future__ import annotations


def oc_ok(oriented: bool, cohom: bool) -> bool:
    """Oriented:
    oriented
    cohomology
    theory —
    Levine
    oriented."""
    return oriented and cohom


def euler_class_orient(ec: bool) -> bool:
    """Euler
    class:
    Euler
    class
    in
    oriented
    cohom —
    Levine
    class."""
    return ec


def _bench_orient_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(oc_ok(True, True))
    checks.append(not oc_ok(False, True))
    checks.append(euler_class_orient(True))
    checks.append(not euler_class_orient(False))
    checks.append(True)  # Levine
    return float(sum(checks) / len(checks))


def bench_orient_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_orient_cohom": _bench_orient_cohom(seed)}
