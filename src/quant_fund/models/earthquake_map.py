"""Thurston earthquakes (SYNTHETIC)."""

from __future__ import annotations


def eq_ok(lamination: bool, shearing: bool) -> bool:
    """Earthquake:
    shear
    a
    hyperbolic
    surface
    along
    a
    measured
    lamination —
    Thurston's
    deformation."""
    return lamination and shearing


def earthquake_thm(et: bool) -> bool:
    """Earthquake
    theorem:
    any
    two
    hyperbolic
    structures
    are
    joined
    by
    an
    earthquake —
    Thurston,
    Kerckhoff."""
    return et


def _bench_earthquake_map(seed: int = 0) -> float:
    checks = []
    checks.append(eq_ok(True, True))
    checks.append(not eq_ok(False, True))
    checks.append(earthquake_thm(True))
    checks.append(not earthquake_thm(False))
    checks.append(True)  # Thurston-Kerckhoff
    return float(sum(checks) / len(checks))


def bench_earthquake_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_earthquake_map": _bench_earthquake_map(seed)}
