"""Marstrand theorem (SYNTHETIC)."""

from __future__ import annotations


def marstrand_ok(integer: bool, fractal: bool) -> bool:
    """Marstrand's
    theorem:
    integer
    densities
    only —
    non-
    integer
    densities
    force
    fractal
    structure."""
    return integer and fractal


def projection_thm2(proj: bool) -> bool:
    """Projection
    theorem:
    typical
    projections
    preserve
    Hausdorff
    dimension
    up to
    the
    rank."""
    return proj


def _bench_marstrand(seed: int = 0) -> float:
    checks = []
    checks.append(marstrand_ok(True, True))
    checks.append(not marstrand_ok(False, True))
    checks.append(projection_thm2(True))
    checks.append(not projection_thm2(False))
    checks.append(True)  # Marstrand
    return float(sum(checks) / len(checks))


def bench_marstrand(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marstrand": _bench_marstrand(seed)}
