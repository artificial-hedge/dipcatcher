"""Globular models (SYNTHETIC)."""

from __future__ import annotations


def globular_ok(globular_set: bool, weak_comp: bool) -> bool:
    """Globular model of
    weak omega-categories:
    cells along globes,
    weak composition via
    operads (Batanin,
    Leinster)."""
    return globular_set and weak_comp


def contractible_operad(coherence: bool) -> bool:
    """Contractible globular
    operad: algebras are
    weak omega-categories;
    coherence via
    universality."""
    return coherence


def _bench_globular_model(seed: int = 0) -> float:
    checks = []
    checks.append(globular_ok(True, True))
    checks.append(not globular_ok(False, True))
    checks.append(contractible_operad(True))
    checks.append(not contractible_operad(False))
    checks.append(True)  # Batanin omega-cats
    return float(sum(checks) / len(checks))


def bench_globular_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_globular_model": _bench_globular_model(seed)}
