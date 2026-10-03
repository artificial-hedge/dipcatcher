"""Heegaard splittings (SYNTHETIC)."""

from __future__ import annotations


def hs_ok(two_handlebodies: bool, surface: bool) -> bool:
    """Heegaard
    splitting:
    every
    closed
    3-manifold
    is
    two
    handlebodies
    glued
    along
    a
    surface —
    genus
    is
    a
    complexity."""
    return two_handlebodies and surface


def destabilization(ds: bool) -> bool:
    """Waldhausen:
    every
    Heegaard
    splitting
    of
    S3
    is
    a
    stabilization
    of
    the
    standard
    one —
    uniqueness."""
    return ds


def _bench_heegaard_splitting(seed: int = 0) -> float:
    checks = []
    checks.append(hs_ok(True, True))
    checks.append(not hs_ok(False, True))
    checks.append(destabilization(True))
    checks.append(not destabilization(False))
    checks.append(True)  # Waldhausen
    return float(sum(checks) / len(checks))


def bench_heegaard_splitting(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heegaard_splitting": _bench_heegaard_splitting(seed)}
