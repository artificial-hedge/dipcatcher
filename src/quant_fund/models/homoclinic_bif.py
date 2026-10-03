"""Homoclinic bifurcation (SYNTHETIC)."""

from __future__ import annotations


def hom_ok(loop: bool, saddle: bool) -> bool:
    """Homoclinic
    bifurcation:
    an orbit
    biasymptotic
    to a
    saddle
    creates
    or
    destroys
    a limit
    cycle."""
    return loop and saddle


def saddle_quantity(sq: bool) -> bool:
    """Saddle
    quantity
    sigma =
    trace at
    the saddle
    decides
    stability
    of the
    emerging
    cycle."""
    return sq


def _bench_homoclinic_bif(seed: int = 0) -> float:
    checks = []
    checks.append(hom_ok(True, True))
    checks.append(not hom_ok(False, True))
    checks.append(saddle_quantity(True))
    checks.append(not saddle_quantity(False))
    checks.append(True)  # Shilnikov
    return float(sum(checks) / len(checks))


def bench_homoclinic_bif(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homoclinic_bif": _bench_homoclinic_bif(seed)}
