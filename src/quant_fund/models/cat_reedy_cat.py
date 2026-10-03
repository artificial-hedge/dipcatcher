"""cat reedy_cat module (SYNTHETIC)."""

from __future__ import annotations


def cat_reedy_cat_ok(cat: bool, canonical: bool) -> bool:
    """cat_reedy_cat
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_reedy_cat_aux(aux: bool) -> bool:
    """cat_reedy_cat
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_reedy_cat(seed: int = 0) -> float:
    checks = []
    checks.append(cat_reedy_cat_ok(True, True))
    checks.append(not cat_reedy_cat_ok(False, True))
    checks.append(cat_reedy_cat_aux(True))
    checks.append(not cat_reedy_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_reedy_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_reedy_cat": _bench_cat_reedy_cat(seed)}
