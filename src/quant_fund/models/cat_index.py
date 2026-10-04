"""cat index module (SYNTHETIC)."""

from __future__ import annotations


def cat_index_ok(category: bool, structure: bool) -> bool:
    """cat_index
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_index_aux(aux: bool) -> bool:
    """cat_index
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_index(seed: int = 0) -> float:
    checks = []
    checks.append(cat_index_ok(True, True))
    checks.append(not cat_index_ok(False, True))
    checks.append(cat_index_aux(True))
    checks.append(not cat_index_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_index": _bench_cat_index(seed)}
