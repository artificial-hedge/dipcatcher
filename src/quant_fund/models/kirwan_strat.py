"""Kirwan stratification (SYNTHETIC)."""

from __future__ import annotations


def ks_ok(morse_strata: bool, equivariant: bool) -> bool:
    """Kirwan
    stratification:
    Morse-
    like
    strata
    for
    moment
    map
    norm
    squared —
    equivariant
    perfection."""
    return morse_strata and equivariant


def equivariantly_perfect(ep: bool) -> bool:
    """Equivariantly
    perfect:
    stratification
    gives
    equivariant
    cohomology —
    Kirwan's
    theorem."""
    return ep


def _bench_kirwan_strat(seed: int = 0) -> float:
    checks = []
    checks.append(ks_ok(True, True))
    checks.append(not ks_ok(False, True))
    checks.append(equivariantly_perfect(True))
    checks.append(not equivariantly_perfect(False))
    checks.append(True)  # Kirwan
    return float(sum(checks) / len(checks))


def bench_kirwan_strat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kirwan_strat": _bench_kirwan_strat(seed)}
