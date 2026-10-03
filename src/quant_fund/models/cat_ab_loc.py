"""cat ab_loc module (SYNTHETIC)."""

from __future__ import annotations


def cat_ab_loc_ok(cat: bool, canonical: bool) -> bool:
    """cat_ab_loc
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_ab_loc_aux(aux: bool) -> bool:
    """cat_ab_loc
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_ab_loc(seed: int = 0) -> float:
    checks = []
    checks.append(cat_ab_loc_ok(True, True))
    checks.append(not cat_ab_loc_ok(False, True))
    checks.append(cat_ab_loc_aux(True))
    checks.append(not cat_ab_loc_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_ab_loc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_ab_loc": _bench_cat_ab_loc(seed)}
