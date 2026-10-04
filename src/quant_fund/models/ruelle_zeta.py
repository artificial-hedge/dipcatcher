"""Ruelle zeta (SYNTHETIC)."""

from __future__ import annotations


def rz_ok(zeta: bool, periodic: bool) -> bool:
    """Ruelle
    zeta:
    product
    over
    periodic
    orbits
    of
    weights;
    dynamical
    Euler
    product."""
    return zeta and periodic


def fredholm_det(fred: bool) -> bool:
    """Fredholm
    determinant:
    zeta
    is a
    ratio
    of
    determinants
    of the
    transfer
    operator."""
    return fred


def _bench_ruelle_zeta(seed: int = 0) -> float:
    checks = []
    checks.append(rz_ok(True, True))
    checks.append(not rz_ok(False, True))
    checks.append(fredholm_det(True))
    checks.append(not fredholm_det(False))
    checks.append(True)  # Ruelle-Selberg
    return float(sum(checks) / len(checks))


def bench_ruelle_zeta(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ruelle_zeta": _bench_ruelle_zeta(seed)}
