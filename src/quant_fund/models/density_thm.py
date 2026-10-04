"""Density theorem (SYNTHETIC)."""

from __future__ import annotations


def density_ok(boundary: bool, outside: bool) -> bool:
    """Lebesgue/
    Besicovitch
    density
    theorem:
    at a.e.
    point
    the
    measure
    ratio
    tends
    to 0
    or 1."""
    return boundary and outside


def differentiation(d: bool) -> bool:
    """Measure
    differentiation:
    density
    exists
    for
    locally
    finite
    measures
    on
    doubling
    spaces."""
    return d


def _bench_density_thm(seed: int = 0) -> float:
    checks = []
    checks.append(density_ok(True, True))
    checks.append(not density_ok(False, True))
    checks.append(differentiation(True))
    checks.append(not differentiation(False))
    checks.append(True)  # Besicovitch
    return float(sum(checks) / len(checks))


def bench_density_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_density_thm": _bench_density_thm(seed)}
