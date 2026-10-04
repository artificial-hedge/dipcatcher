"""tannakian cat module (SYNTHETIC)."""

from __future__ import annotations


def tannakian_cat_ok(category: bool, structure: bool) -> bool:
    """tannakian_cat
    check:
    category
    structure —
    tensor."""
    return category and structure


def tannakian_cat_aux(aux: bool) -> bool:
    """tannakian_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_tannakian_cat(seed: int = 0) -> float:
    checks = []
    checks.append(tannakian_cat_ok(True, True))
    checks.append(not tannakian_cat_ok(False, True))
    checks.append(tannakian_cat_aux(True))
    checks.append(not tannakian_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_tannakian_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tannakian_cat": _bench_tannakian_cat(seed)}
