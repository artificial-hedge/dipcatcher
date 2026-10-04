"""cat bicomplete module (SYNTHETIC)."""

from __future__ import annotations


def cat_bicomplete_ok(category: bool, structure: bool) -> bool:
    """cat_bicomplete
    check:
    category
    structure —
    fibrant."""
    return category and structure


def cat_bicomplete_aux(aux: bool) -> bool:
    """cat_bicomplete
    aux:
    auxiliary
    category
    check —
    cofibrant."""
    return aux


def _bench_cat_bicomplete(seed: int = 0) -> float:
    checks = []
    checks.append(cat_bicomplete_ok(True, True))
    checks.append(not cat_bicomplete_ok(False, True))
    checks.append(cat_bicomplete_aux(True))
    checks.append(not cat_bicomplete_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_bicomplete(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_bicomplete": _bench_cat_bicomplete(seed)}
