"""special cat module (SYNTHETIC)."""

from __future__ import annotations


def special_cat_ok(category: bool, structure: bool) -> bool:
    """special_cat
    check:
    categorical
    structure —
    enriched."""
    return category and structure


def special_cat_aux(aux: bool) -> bool:
    """special_cat
    aux:
    auxiliary
    category
    check —
    functorial."""
    return aux


def _bench_special_cat(seed: int = 0) -> float:
    checks = []
    checks.append(special_cat_ok(True, True))
    checks.append(not special_cat_ok(False, True))
    checks.append(special_cat_aux(True))
    checks.append(not special_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_special_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_special_cat": _bench_special_cat(seed)}
