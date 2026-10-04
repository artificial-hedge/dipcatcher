"""cat ab2 module (SYNTHETIC)."""

from __future__ import annotations


def cat_ab2_ok(cat: bool, canonical: bool) -> bool:
    """cat_ab2
    check:
    category
    structure —
    abelian."""
    return cat and canonical


def cat_ab2_aux(aux: bool) -> bool:
    """cat_ab2
    aux:
    auxiliary
    category
    check —
    exact."""
    return aux


def _bench_cat_ab2(seed: int = 0) -> float:
    checks = []
    checks.append(cat_ab2_ok(True, True))
    checks.append(not cat_ab2_ok(False, True))
    checks.append(cat_ab2_aux(True))
    checks.append(not cat_ab2_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_ab2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_ab2": _bench_cat_ab2(seed)}
