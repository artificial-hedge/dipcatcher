"""cat span module (SYNTHETIC)."""

from __future__ import annotations


def cat_span_ok(category: bool, structure: bool) -> bool:
    """cat_span
    check:
    category
    structure —
    pushout."""
    return category and structure


def cat_span_aux(aux: bool) -> bool:
    """cat_span
    aux:
    auxiliary
    category
    check —
    span."""
    return aux


def _bench_cat_span(seed: int = 0) -> float:
    checks = []
    checks.append(cat_span_ok(True, True))
    checks.append(not cat_span_ok(False, True))
    checks.append(cat_span_aux(True))
    checks.append(not cat_span_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_span(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_span": _bench_cat_span(seed)}
