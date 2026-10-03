"""cat size module (SYNTHETIC)."""

from __future__ import annotations


def cat_size_ok(category: bool, structure: bool) -> bool:
    """cat_size
    check:
    category
    structure —
    pushout."""
    return category and structure


def cat_size_aux(aux: bool) -> bool:
    """cat_size
    aux:
    auxiliary
    category
    check —
    span."""
    return aux


def _bench_cat_size(seed: int = 0) -> float:
    checks = []
    checks.append(cat_size_ok(True, True))
    checks.append(not cat_size_ok(False, True))
    checks.append(cat_size_aux(True))
    checks.append(not cat_size_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_size(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_size": _bench_cat_size(seed)}
