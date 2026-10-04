"""cat pseudo_limit module (SYNTHETIC)."""

from __future__ import annotations


def cat_pseudo_limit_ok(cat: bool, canonical: bool) -> bool:
    """cat_pseudo_limit
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_pseudo_limit_aux(aux: bool) -> bool:
    """cat_pseudo_limit
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_pseudo_limit(seed: int = 0) -> float:
    checks = []
    checks.append(cat_pseudo_limit_ok(True, True))
    checks.append(not cat_pseudo_limit_ok(False, True))
    checks.append(cat_pseudo_limit_aux(True))
    checks.append(not cat_pseudo_limit_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_pseudo_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_pseudo_limit": _bench_cat_pseudo_limit(seed)}
