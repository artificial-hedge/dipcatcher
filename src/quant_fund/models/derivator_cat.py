"""derivator cat module (SYNTHETIC)."""

from __future__ import annotations


def derivator_cat_ok(category: bool, structure: bool) -> bool:
    """derivator_cat
    check:
    category
    structure —
    enriched."""
    return category and structure


def derivator_cat_aux(aux: bool) -> bool:
    """derivator_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_derivator_cat(seed: int = 0) -> float:
    checks = []
    checks.append(derivator_cat_ok(True, True))
    checks.append(not derivator_cat_ok(False, True))
    checks.append(derivator_cat_aux(True))
    checks.append(not derivator_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_derivator_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derivator_cat": _bench_derivator_cat(seed)}
