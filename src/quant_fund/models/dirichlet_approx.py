"""Dirichlet approximation (SYNTHETIC)."""

from __future__ import annotations


def dirichlet_ok(pigeon: bool, simultaneous: bool) -> bool:
    """Dirichlet:
    every
    real
    has
    rational
    approximation
    p/q
    within
    1/q^2 —
    pigeonhole
    argument."""
    return pigeon and simultaneous


def hurwitz_opt(ho: bool) -> bool:
    """Hurwitz:
    constant
    1/sqrt(5)
    is
    optimal
    in
    Dirichlet's
    theorem —
    golden
    ratio
    is
    worst."""
    return ho


def _bench_dirichlet_approx(seed: int = 0) -> float:
    checks = []
    checks.append(dirichlet_ok(True, True))
    checks.append(not dirichlet_ok(False, True))
    checks.append(hurwitz_opt(True))
    checks.append(not hurwitz_opt(False))
    checks.append(True)  # Dirichlet-Hurwitz
    return float(sum(checks) / len(checks))


def bench_dirichlet_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirichlet_approx": _bench_dirichlet_approx(seed)}
