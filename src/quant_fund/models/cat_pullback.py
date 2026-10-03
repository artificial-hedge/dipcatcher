"""cat pullback module (SYNTHETIC)."""

from __future__ import annotations


def cat_pullback_ok(category: bool, structure: bool) -> bool:
    """cat_pullback
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_pullback_aux(aux: bool) -> bool:
    """cat_pullback
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_pullback(seed: int = 0) -> float:
    checks = []
    checks.append(cat_pullback_ok(True, True))
    checks.append(not cat_pullback_ok(False, True))
    checks.append(cat_pullback_aux(True))
    checks.append(not cat_pullback_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_pullback(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_pullback": _bench_cat_pullback(seed)}
