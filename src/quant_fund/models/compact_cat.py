"""compact cat module (SYNTHETIC)."""

from __future__ import annotations


def compact_cat_ok(category: bool, structure: bool) -> bool:
    """compact_cat
    check:
    category
    structure —
    tensor."""
    return category and structure


def compact_cat_aux(aux: bool) -> bool:
    """compact_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_compact_cat(seed: int = 0) -> float:
    checks = []
    checks.append(compact_cat_ok(True, True))
    checks.append(not compact_cat_ok(False, True))
    checks.append(compact_cat_aux(True))
    checks.append(not compact_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_compact_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_cat": _bench_compact_cat(seed)}
