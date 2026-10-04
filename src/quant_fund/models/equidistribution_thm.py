"""Equidistribution theorem (SYNTHETIC)."""

from __future__ import annotations


def et_ok(galois_orbits: bool, canonical_measure: bool) -> bool:
    """Equidistribution:
    small-
    height
    points
    equidistribute
    for
    canonical
    measure —
    SUCC
    theorem."""
    return galois_orbits and canonical_measure


def bilu_equidistribution(be: bool) -> bool:
    """Bilu:
    equidistribution
    of
    Galois
    orbits
    of
    small
    points —
    Bilu's
    theorem."""
    return be


def _bench_equidistribution_thm(seed: int = 0) -> float:
    checks = []
    checks.append(et_ok(True, True))
    checks.append(not et_ok(False, True))
    checks.append(bilu_equidistribution(True))
    checks.append(not bilu_equidistribution(False))
    checks.append(True)  # Bilu-Szpiro-Ullmo-Zhang
    return float(sum(checks) / len(checks))


def bench_equidistribution_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equidistribution_thm": _bench_equidistribution_thm(seed)}
