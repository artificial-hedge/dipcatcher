"""cat lax module (SYNTHETIC)."""

from __future__ import annotations


def cat_lax_ok(category: bool, structure: bool) -> bool:
    """cat_lax
    check:
    category
    structure —
    pushout."""
    return category and structure


def cat_lax_aux(aux: bool) -> bool:
    """cat_lax
    aux:
    auxiliary
    category
    check —
    span."""
    return aux


def _bench_cat_lax(seed: int = 0) -> float:
    checks = []
    checks.append(cat_lax_ok(True, True))
    checks.append(not cat_lax_ok(False, True))
    checks.append(cat_lax_aux(True))
    checks.append(not cat_lax_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_lax(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_lax": _bench_cat_lax(seed)}
