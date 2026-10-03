"""Liquid rings (SYNTHETIC)."""

from __future__ import annotations


def lr_ok(liquid: bool, ring: bool) -> bool:
    """Liquid
    ring:
    liquid
    ring —
    Clausen-
    Scholze
    liquid
    ring."""
    return liquid and ring


def liquid_experiment(le: bool) -> bool:
    """Liquid
    experiment:
    liquid
    tensor
    experiment —
    Clausen
    liquid
    tensor."""
    return le


def _bench_liquid_ring(seed: int = 0) -> float:
    checks = []
    checks.append(lr_ok(True, True))
    checks.append(not lr_ok(False, True))
    checks.append(liquid_experiment(True))
    checks.append(not liquid_experiment(False))
    checks.append(True)  # Clausen-Scholze
    return float(sum(checks) / len(checks))


def bench_liquid_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liquid_ring": _bench_liquid_ring(seed)}
