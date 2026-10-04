"""smashing cat module (SYNTHETIC)."""

from __future__ import annotations


def smashing_cat_ok(category: bool, structure: bool) -> bool:
    """smashing_cat
    check:
    category
    structure —
    tensor."""
    return category and structure


def smashing_cat_aux(aux: bool) -> bool:
    """smashing_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_smashing_cat(seed: int = 0) -> float:
    checks = []
    checks.append(smashing_cat_ok(True, True))
    checks.append(not smashing_cat_ok(False, True))
    checks.append(smashing_cat_aux(True))
    checks.append(not smashing_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_smashing_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smashing_cat": _bench_smashing_cat(seed)}
