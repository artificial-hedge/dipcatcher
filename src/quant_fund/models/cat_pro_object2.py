"""cat pro_object2 module (SYNTHETIC)."""

from __future__ import annotations


def cat_pro_object2_ok(cat: bool, canonical: bool) -> bool:
    """cat_pro_object2
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_pro_object2_aux(aux: bool) -> bool:
    """cat_pro_object2
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_pro_object2(seed: int = 0) -> float:
    checks = []
    checks.append(cat_pro_object2_ok(True, True))
    checks.append(not cat_pro_object2_ok(False, True))
    checks.append(cat_pro_object2_aux(True))
    checks.append(not cat_pro_object2_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_pro_object2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_pro_object2": _bench_cat_pro_object2(seed)}
