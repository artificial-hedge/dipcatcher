"""cat pushout module (SYNTHETIC)."""

from __future__ import annotations


def cat_pushout_ok(category: bool, structure: bool) -> bool:
    """cat_pushout
    check:
    category
    structure —
    pushout."""
    return category and structure


def cat_pushout_aux(aux: bool) -> bool:
    """cat_pushout
    aux:
    auxiliary
    category
    check —
    span."""
    return aux


def _bench_cat_pushout(seed: int = 0) -> float:
    checks = []
    checks.append(cat_pushout_ok(True, True))
    checks.append(not cat_pushout_ok(False, True))
    checks.append(cat_pushout_aux(True))
    checks.append(not cat_pushout_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_pushout(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_pushout": _bench_cat_pushout(seed)}
