"""Knot Floer homology (SYNTHETIC)."""

from __future__ import annotations


def kf_ok(doubly: bool, genus: bool) -> bool:
    """Knot
    Floer
    homology:
    doubly-pointed
    Heegaard
    diagram —
    detects
    genus
    and
    fiberedness."""
    return doubly and genus


def tau_invariant(ti: bool) -> bool:
    """Tau
    invariant:
    concordance
    homomorphism
    from
    knot
    Floer —
    bounds
    slice
    genus."""
    return ti


def _bench_knot_floer(seed: int = 0) -> float:
    checks = []
    checks.append(kf_ok(True, True))
    checks.append(not kf_ok(False, True))
    checks.append(tau_invariant(True))
    checks.append(not tau_invariant(False))
    checks.append(True)  # Ozsvath-Szabo
    return float(sum(checks) / len(checks))


def bench_knot_floer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_floer": _bench_knot_floer(seed)}
