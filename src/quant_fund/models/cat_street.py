"""cat street module (SYNTHETIC)."""

from __future__ import annotations


def cat_street_ok(category: bool, structure: bool) -> bool:
    """cat_street
    check:
    category
    structure —
    pushout."""
    return category and structure


def cat_street_aux(aux: bool) -> bool:
    """cat_street
    aux:
    auxiliary
    category
    check —
    span."""
    return aux


def _bench_cat_street(seed: int = 0) -> float:
    checks = []
    checks.append(cat_street_ok(True, True))
    checks.append(not cat_street_ok(False, True))
    checks.append(cat_street_aux(True))
    checks.append(not cat_street_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_street(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_street": _bench_cat_street(seed)}
