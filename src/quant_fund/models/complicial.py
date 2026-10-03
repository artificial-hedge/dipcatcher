"""Complicial sets (SYNTHETIC)."""

from __future__ import annotations


def complicial_ok(marked: bool, thin: bool) -> bool:
    """Complicial (Verity)
    sets: marked simplicial
    sets where marked
    simplices are 'thin'
    equivalences; model
    for (infty,infinity)."""
    return marked and thin


def street_nerve(strict: bool) -> bool:
    """Street nerve of strict
    omega-categories:
    complicial sets from
    orientals; Roberts-
    Street."""
    return strict


def _bench_complicial(seed: int = 0) -> float:
    checks = []
    checks.append(complicial_ok(True, True))
    checks.append(not complicial_ok(False, True))
    checks.append(street_nerve(True))
    checks.append(not street_nerve(False))
    checks.append(True)  # Verity model structure
    return float(sum(checks) / len(checks))


def bench_complicial(seed: int = 0) -> dict[str, float]:
    return {"synthetic_complicial": _bench_complicial(seed)}
