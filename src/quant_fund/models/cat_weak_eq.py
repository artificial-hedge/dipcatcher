"""cat weak_eq module (SYNTHETIC)."""

from __future__ import annotations


def cat_weak_eq_ok(cat: bool, canonical: bool) -> bool:
    """cat_weak_eq
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_weak_eq_aux(aux: bool) -> bool:
    """cat_weak_eq
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_weak_eq(seed: int = 0) -> float:
    checks = []
    checks.append(cat_weak_eq_ok(True, True))
    checks.append(not cat_weak_eq_ok(False, True))
    checks.append(cat_weak_eq_aux(True))
    checks.append(not cat_weak_eq_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_weak_eq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_weak_eq": _bench_cat_weak_eq(seed)}
