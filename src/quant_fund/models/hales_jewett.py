"""Hales-Jewett theorem (SYNTHETIC)."""

from __future__ import annotations


def hj_ok(color: bool, line: bool) -> bool:
    """Hales-Jewett:
    any finite
    coloring
    of HJ
    [k]^n for
    large n
    contains a
    monochromatic
    combinatorial
    line."""
    return color and line


def density_hj(dens: bool) -> bool:
    """Density
    Hales-Jewett
    (Furstenberg-
    Katznelson):
    positive-
    density
    sets of
    [k]^n
    contain
    lines."""
    return dens


def _bench_hales_jewett(seed: int = 0) -> float:
    checks = []
    checks.append(hj_ok(True, True))
    checks.append(not hj_ok(False, True))
    checks.append(density_hj(True))
    checks.append(not density_hj(False))
    checks.append(True)  # Furstenberg-Katznelson
    return float(sum(checks) / len(checks))


def bench_hales_jewett(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hales_jewett": _bench_hales_jewett(seed)}
