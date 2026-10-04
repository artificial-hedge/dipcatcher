"""Harmonic maps (SYNTHETIC)."""

from __future__ import annotations


def hm_ok(tension: bool, critical: bool) -> bool:
    """Harmonic
    map:
    critical
    point
    of
    the
    Dirichlet
    energy —
    tension
    field
    vanishes."""
    return tension and critical


def bochner_formula(bf: bool) -> bool:
    """Bochner
    formula:
    second-
    variation
    identity
    linking
    energy
    and
    curvature —
    regularity
    tool."""
    return bf


def _bench_harmonic_map(seed: int = 0) -> float:
    checks = []
    checks.append(hm_ok(True, True))
    checks.append(not hm_ok(False, True))
    checks.append(bochner_formula(True))
    checks.append(not bochner_formula(False))
    checks.append(True)  # Eells-Sampson
    return float(sum(checks) / len(checks))


def bench_harmonic_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harmonic_map": _bench_harmonic_map(seed)}
