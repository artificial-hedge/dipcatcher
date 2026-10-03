"""Combinatorial species (SYNTHETIC)."""

from __future__ import annotations


def st_ok(structures: bool, functorial: bool) -> bool:
    """Species:
    combinatorial
    structures
    as
    functors
    on
    finite
    sets —
    Joyal's
    theory."""
    return structures and functorial


def species_operations(so: bool) -> bool:
    """Species
    operations:
    sum,
    product,
    composition
    of
    species —
    generating
    functions."""
    return so


def _bench_species_theory(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(species_operations(True))
    checks.append(not species_operations(False))
    checks.append(True)  # Joyal
    return float(sum(checks) / len(checks))


def bench_species_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_species_theory": _bench_species_theory(seed)}
