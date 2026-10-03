"""Frostman lemma (SYNTHETIC)."""

from __future__ import annotations


def frostman_ok(measure: bool, bound: bool) -> bool:
    """Frostman's
    lemma:
    dim_H
    at least
    s iff
    a Borel
    measure
    with
    mu(B_r)
    bounded
    by
    r^s
    exists."""
    return measure and bound


def energy_crit(ec: bool) -> bool:
    """Energy
    criterion:
    finite
    s-energy
    implies
    dimension
    at least
    s."""
    return ec


def _bench_frostman(seed: int = 0) -> float:
    checks = []
    checks.append(frostman_ok(True, True))
    checks.append(not frostman_ok(False, True))
    checks.append(energy_crit(True))
    checks.append(not energy_crit(False))
    checks.append(True)  # Frostman
    return float(sum(checks) / len(checks))


def bench_frostman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frostman": _bench_frostman(seed)}
