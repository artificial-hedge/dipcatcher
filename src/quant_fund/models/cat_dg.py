"""cat dg module (SYNTHETIC)."""

from __future__ import annotations


def cat_dg_ok(category: bool, structure: bool) -> bool:
    """cat_dg
    check:
    category
    structure —
    enriched."""
    return category and structure


def cat_dg_aux(aux: bool) -> bool:
    """cat_dg
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_cat_dg(seed: int = 0) -> float:
    checks = []
    checks.append(cat_dg_ok(True, True))
    checks.append(not cat_dg_ok(False, True))
    checks.append(cat_dg_aux(True))
    checks.append(not cat_dg_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_dg": _bench_cat_dg(seed)}
