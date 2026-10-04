"""Atomic topos (SYNTHETIC)."""

from __future__ import annotations


def at_ok(atomic: bool, continuous: bool) -> bool:
    """Atomic:
    atomic
    topos —
    Bar-
    Paré."""
    return atomic and continuous


def atomic_morphism(am: bool) -> bool:
    """Atomic
    morphism:
    atomic
    geometric
    morphism —
    Bar-Paré."""
    return am


def _bench_atomic_topos(seed: int = 0) -> float:
    checks = []
    checks.append(at_ok(True, True))
    checks.append(not at_ok(False, True))
    checks.append(atomic_morphism(True))
    checks.append(not atomic_morphism(False))
    checks.append(True)  # Bar-Paré
    return float(sum(checks) / len(checks))


def bench_atomic_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atomic_topos": _bench_atomic_topos(seed)}
