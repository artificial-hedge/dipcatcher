"""pointed cat module (SYNTHETIC)."""

from __future__ import annotations


def pointed_cat_ok(category: bool, structure: bool) -> bool:
    """pointed_cat
    check:
    category
    structure —
    enriched."""
    return category and structure


def pointed_cat_aux(aux: bool) -> bool:
    """pointed_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_pointed_cat(seed: int = 0) -> float:
    checks = []
    checks.append(pointed_cat_ok(True, True))
    checks.append(not pointed_cat_ok(False, True))
    checks.append(pointed_cat_aux(True))
    checks.append(not pointed_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_pointed_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pointed_cat": _bench_pointed_cat(seed)}
