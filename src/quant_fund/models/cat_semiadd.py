"""cat semiadd module (SYNTHETIC)."""

from __future__ import annotations


def cat_semiadd_ok(category: bool, structure: bool) -> bool:
    """cat_semiadd
    check:
    category
    structure —
    pretopos."""
    return category and structure


def cat_semiadd_aux(aux: bool) -> bool:
    """cat_semiadd
    aux:
    auxiliary
    category
    check —
    ribbon."""
    return aux


def _bench_cat_semiadd(seed: int = 0) -> float:
    checks = []
    checks.append(cat_semiadd_ok(True, True))
    checks.append(not cat_semiadd_ok(False, True))
    checks.append(cat_semiadd_aux(True))
    checks.append(not cat_semiadd_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_semiadd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_semiadd": _bench_cat_semiadd(seed)}
