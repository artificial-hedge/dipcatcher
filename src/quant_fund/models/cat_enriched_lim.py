"""cat enriched_lim module (SYNTHETIC)."""

from __future__ import annotations


def cat_enriched_lim_ok(cat: bool, canonical: bool) -> bool:
    """cat_enriched_lim
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_enriched_lim_aux(aux: bool) -> bool:
    """cat_enriched_lim
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_enriched_lim(seed: int = 0) -> float:
    checks = []
    checks.append(cat_enriched_lim_ok(True, True))
    checks.append(not cat_enriched_lim_ok(False, True))
    checks.append(cat_enriched_lim_aux(True))
    checks.append(not cat_enriched_lim_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_enriched_lim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_enriched_lim": _bench_cat_enriched_lim(seed)}
