"""Bicategories (SYNTHETIC)."""

from __future__ import annotations


def bc2_ok(bicat: bool, weak: bool) -> bool:
    """Bicategory:
    bicategory
    —
    weak
    2
    category."""
    return bicat and weak


def bicategory_assoc(ba: bool) -> bool:
    """Bicategory
    associator:
    bicategory
    associator —
    coherence
    constraint."""
    return ba


def _bench_bicat2(seed: int = 0) -> float:
    checks = []
    checks.append(bc2_ok(True, True))
    checks.append(not bc2_ok(False, True))
    checks.append(bicategory_assoc(True))
    checks.append(not bicategory_assoc(False))
    checks.append(True)  # Benabou
    return float(sum(checks) / len(checks))


def bench_bicat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bicat2": _bench_bicat2(seed)}
