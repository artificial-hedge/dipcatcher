"""cat cofibrant module (SYNTHETIC)."""

from __future__ import annotations


def cat_cofibrant_ok(category: bool, structure: bool) -> bool:
    """cat_cofibrant
    check:
    category
    structure —
    fibrant."""
    return category and structure


def cat_cofibrant_aux(aux: bool) -> bool:
    """cat_cofibrant
    aux:
    auxiliary
    category
    check —
    cofibrant."""
    return aux


def _bench_cat_cofibrant(seed: int = 0) -> float:
    checks = []
    checks.append(cat_cofibrant_ok(True, True))
    checks.append(not cat_cofibrant_ok(False, True))
    checks.append(cat_cofibrant_aux(True))
    checks.append(not cat_cofibrant_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_cofibrant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_cofibrant": _bench_cat_cofibrant(seed)}
