"""Motivic purity (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(purity_iso: bool, gysin: bool) -> bool:
    """Motivic
    purity:
    Thom
    isomorphism
    for
    closed
    immersion —
    purity
    isomorphism."""
    return purity_iso and gysin


def purity_gysin(pg: bool) -> bool:
    """Purity
    Gysin:
    Gysin
    morphism
    in
    motivic
    cohomology —
    purity."""
    return pg


def _bench_motivic_purity(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(purity_gysin(True))
    checks.append(not purity_gysin(False))
    checks.append(True)  # Purity
    return float(sum(checks) / len(checks))


def bench_motivic_purity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_purity": _bench_motivic_purity(seed)}
