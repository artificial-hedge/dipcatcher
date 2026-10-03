"""relative cat module (SYNTHETIC)."""

from __future__ import annotations


def relative_cat_ok(category: bool, structure: bool) -> bool:
    """relative_cat
    check:
    category
    structure —
    enriched."""
    return category and structure


def relative_cat_aux(aux: bool) -> bool:
    """relative_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_relative_cat(seed: int = 0) -> float:
    checks = []
    checks.append(relative_cat_ok(True, True))
    checks.append(not relative_cat_ok(False, True))
    checks.append(relative_cat_aux(True))
    checks.append(not relative_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_relative_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_relative_cat": _bench_relative_cat(seed)}
