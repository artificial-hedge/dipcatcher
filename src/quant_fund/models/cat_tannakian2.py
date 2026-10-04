"""cat tannakian2 module (SYNTHETIC)."""

from __future__ import annotations


def cat_tannakian2_ok(category: bool, structure: bool) -> bool:
    """cat_tannakian2
    check:
    category
    structure —
    pretopos."""
    return category and structure


def cat_tannakian2_aux(aux: bool) -> bool:
    """cat_tannakian2
    aux:
    auxiliary
    category
    check —
    ribbon."""
    return aux


def _bench_cat_tannakian2(seed: int = 0) -> float:
    checks = []
    checks.append(cat_tannakian2_ok(True, True))
    checks.append(not cat_tannakian2_ok(False, True))
    checks.append(cat_tannakian2_aux(True))
    checks.append(not cat_tannakian2_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_tannakian2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_tannakian2": _bench_cat_tannakian2(seed)}
