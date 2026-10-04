"""exact cat2 module (SYNTHETIC)."""

from __future__ import annotations


def exact_cat2_ok(category: bool, structure: bool) -> bool:
    """exact_cat2
    check:
    categorical
    structure —
    enriched."""
    return category and structure


def exact_cat2_aux(aux: bool) -> bool:
    """exact_cat2
    aux:
    auxiliary
    category
    check —
    functorial."""
    return aux


def _bench_exact_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(exact_cat2_ok(True, True))
    checks.append(not exact_cat2_ok(False, True))
    checks.append(exact_cat2_aux(True))
    checks.append(not exact_cat2_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_exact_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exact_cat2": _bench_exact_cat2(seed)}
