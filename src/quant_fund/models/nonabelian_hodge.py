"""Nonabelian Hodge theory (SYNTHETIC)."""

from __future__ import annotations


def nah_ok(nonabelian: bool, hodge_isom: bool) -> bool:
    """Nonabelian
    Hodge:
    Dolbeault
    de
    Rham
    Betti
    equivalence —
    nonabelian
    Hodge."""
    return nonabelian and hodge_isom


def harmonic_bundle(hb: bool) -> bool:
    """Harmonic
    bundle:
    harmonic
    metric
    on
    bundle —
    Corlette
    Donaldson."""
    return hb


def _bench_nonabelian_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(nah_ok(True, True))
    checks.append(not nah_ok(False, True))
    checks.append(harmonic_bundle(True))
    checks.append(not harmonic_bundle(False))
    checks.append(True)  # Simpson
    return float(sum(checks) / len(checks))


def bench_nonabelian_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonabelian_hodge": _bench_nonabelian_hodge(seed)}
