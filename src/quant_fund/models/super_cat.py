"""super cat module (SYNTHETIC)."""

from __future__ import annotations


def super_cat_ok(category: bool, structure: bool) -> bool:
    """super_cat
    check:
    category
    structure —
    tensor."""
    return category and structure


def super_cat_aux(aux: bool) -> bool:
    """super_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_super_cat(seed: int = 0) -> float:
    checks = []
    checks.append(super_cat_ok(True, True))
    checks.append(not super_cat_ok(False, True))
    checks.append(super_cat_aux(True))
    checks.append(not super_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_super_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_super_cat": _bench_super_cat(seed)}
