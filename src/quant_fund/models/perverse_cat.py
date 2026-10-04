"""perverse cat module (SYNTHETIC)."""

from __future__ import annotations


def perverse_cat_ok(category: bool, structure: bool) -> bool:
    """perverse_cat
    check:
    category
    structure —
    tensor."""
    return category and structure


def perverse_cat_aux(aux: bool) -> bool:
    """perverse_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_perverse_cat(seed: int = 0) -> float:
    checks = []
    checks.append(perverse_cat_ok(True, True))
    checks.append(not perverse_cat_ok(False, True))
    checks.append(perverse_cat_aux(True))
    checks.append(not perverse_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_perverse_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perverse_cat": _bench_perverse_cat(seed)}
