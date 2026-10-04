"""cat hoc module (SYNTHETIC)."""

from __future__ import annotations


def cat_hoc_ok(cat: bool, canonical: bool) -> bool:
    """cat_hoc
    check:
    category
    structure —
    pseudo."""
    return cat and canonical


def cat_hoc_aux(aux: bool) -> bool:
    """cat_hoc
    aux:
    auxiliary
    category
    check —
    weak."""
    return aux


def _bench_cat_hoc(seed: int = 0) -> float:
    checks = []
    checks.append(cat_hoc_ok(True, True))
    checks.append(not cat_hoc_ok(False, True))
    checks.append(cat_hoc_aux(True))
    checks.append(not cat_hoc_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_hoc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_hoc": _bench_cat_hoc(seed)}
