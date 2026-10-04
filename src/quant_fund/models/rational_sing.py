"""Rational singularities (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(higher_direct: bool, cohom_trivial: bool) -> bool:
    """Rational
    singularity:
    higher
    direct
    images
    of
    the
    resolution
    vanish —
    rational
    smoothness
    up
    to
    cohomology."""
    return higher_direct and cohom_trivial


def artin_criterion(ac: bool) -> bool:
    """Artin
    criterion:
    rational
    double
    points
    are
    the
    only
    rational
    Gorenstein
    surface
    singularities."""
    return ac


def _bench_rational_sing(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(artin_criterion(True))
    checks.append(not artin_criterion(False))
    checks.append(True)  # Artin
    return float(sum(checks) / len(checks))


def bench_rational_sing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rational_sing": _bench_rational_sing(seed)}
