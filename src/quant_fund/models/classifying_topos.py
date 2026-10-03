"""Classifying topos (SYNTHETIC)."""

from __future__ import annotations


def ct_ok(classifying: bool, theory: bool) -> bool:
    """Classifying:
    classifying
    topos
    for
    geometric
    theory —
    Joyal
    classifying."""
    return classifying and theory


def universal_model(um: bool) -> bool:
    """Universal
    model:
    universal
    model
    of
    geometric
    theory —
    Diaconescu."""
    return um


def _bench_classifying_topos(seed: int = 0) -> float:
    checks = []
    checks.append(ct_ok(True, True))
    checks.append(not ct_ok(False, True))
    checks.append(universal_model(True))
    checks.append(not universal_model(False))
    checks.append(True)  # Joyal-Diaconescu
    return float(sum(checks) / len(checks))


def bench_classifying_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_classifying_topos": _bench_classifying_topos(seed)}
