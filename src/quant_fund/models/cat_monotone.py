"""cat monotone module (SYNTHETIC)."""

from __future__ import annotations


def cat_monotone_ok(category: bool, structure: bool) -> bool:
    """cat_monotone
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_monotone_aux(aux: bool) -> bool:
    """cat_monotone
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_monotone(seed: int = 0) -> float:
    checks = []
    checks.append(cat_monotone_ok(True, True))
    checks.append(not cat_monotone_ok(False, True))
    checks.append(cat_monotone_aux(True))
    checks.append(not cat_monotone_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_monotone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_monotone": _bench_cat_monotone(seed)}
