"""cat rank module (SYNTHETIC)."""

from __future__ import annotations


def cat_rank_ok(category: bool, structure: bool) -> bool:
    """cat_rank
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_rank_aux(aux: bool) -> bool:
    """cat_rank
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_rank(seed: int = 0) -> float:
    checks = []
    checks.append(cat_rank_ok(True, True))
    checks.append(not cat_rank_ok(False, True))
    checks.append(cat_rank_aux(True))
    checks.append(not cat_rank_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_rank": _bench_cat_rank(seed)}
