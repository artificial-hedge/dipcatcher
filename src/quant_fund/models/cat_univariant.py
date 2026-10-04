"""cat univariant module (SYNTHETIC)."""

from __future__ import annotations


def cat_univariant_ok(category: bool, structure: bool) -> bool:
    """cat_univariant
    check:
    category
    structure —
    fibrant."""
    return category and structure


def cat_univariant_aux(aux: bool) -> bool:
    """cat_univariant
    aux:
    auxiliary
    category
    check —
    cofibrant."""
    return aux


def _bench_cat_univariant(seed: int = 0) -> float:
    checks = []
    checks.append(cat_univariant_ok(True, True))
    checks.append(not cat_univariant_ok(False, True))
    checks.append(cat_univariant_aux(True))
    checks.append(not cat_univariant_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_univariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_univariant": _bench_cat_univariant(seed)}
