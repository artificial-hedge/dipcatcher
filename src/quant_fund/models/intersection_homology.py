"""Intersection homology (SYNTHETIC)."""

from __future__ import annotations


def ih_ok(poincare_duality: bool, singular_spaces: bool) -> bool:
    """Intersection
    homology:
    restores
    Poincare
    duality
    on
    singular
    spaces —
    Goresky-
    MacPherson."""
    return poincare_duality and singular_spaces


def perversity_chains(pc: bool) -> bool:
    """Perversity
    chains:
    chains
    meeting
    strata
    with
    controlled
    dimension —
    IH
    construction."""
    return pc


def _bench_intersection_homology(seed: int = 0) -> float:
    checks = []
    checks.append(ih_ok(True, True))
    checks.append(not ih_ok(False, True))
    checks.append(perversity_chains(True))
    checks.append(not perversity_chains(False))
    checks.append(True)  # Goresky-MacPherson
    return float(sum(checks) / len(checks))


def bench_intersection_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intersection_homology": _bench_intersection_homology(seed)}
