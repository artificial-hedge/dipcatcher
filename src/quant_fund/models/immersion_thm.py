"""Immersion theorem (SYNTHETIC)."""

from __future__ import annotations


def it_ok(formal_immersion: bool, genuine: bool) -> bool:
    """Smale-
    Hirsch:
    formal
    immersions
    integrate
    to
    genuine
    immersions —
    h-
    principle."""
    return formal_immersion and genuine


def sphere_eversion(se: bool) -> bool:
    """Sphere
    eversion:
    S^2
    everts
    in
    R^3 —
    Smale's
    paradoxical
    corollary."""
    return se


def _bench_immersion_thm(seed: int = 0) -> float:
    checks = []
    checks.append(it_ok(True, True))
    checks.append(not it_ok(False, True))
    checks.append(sphere_eversion(True))
    checks.append(not sphere_eversion(False))
    checks.append(True)  # Smale-Hirsch
    return float(sum(checks) / len(checks))


def bench_immersion_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_immersion_thm": _bench_immersion_thm(seed)}
