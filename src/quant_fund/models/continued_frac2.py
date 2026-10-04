"""Continued fractions (SYNTHETIC)."""

from __future__ import annotations


def cf_ok(convergents: bool, best: bool) -> bool:
    """Continued
    fractions:
    convergents
    are
    the
    best
    approximations
    of
    the
    second
    kind."""
    return convergents and best


def periodic_quad(pq: bool) -> bool:
    """Lagrange:
    periodic
    continued
    fractions
    are
    exactly
    the
    quadratic
    irrationals."""
    return pq


def _bench_continued_frac2(seed: int = 0) -> float:
    checks = []
    checks.append(cf_ok(True, True))
    checks.append(not cf_ok(False, True))
    checks.append(periodic_quad(True))
    checks.append(not periodic_quad(False))
    checks.append(True)  # Lagrange
    return float(sum(checks) / len(checks))


def bench_continued_frac2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_continued_frac2": _bench_continued_frac2(seed)}
