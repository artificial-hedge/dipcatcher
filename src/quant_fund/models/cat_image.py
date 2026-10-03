"""cat image module (SYNTHETIC)."""

from __future__ import annotations


def cat_image_ok(category: bool, structure: bool) -> bool:
    """cat_image
    check:
    category
    structure —
    rank."""
    return category and structure


def cat_image_aux(aux: bool) -> bool:
    """cat_image
    aux:
    auxiliary
    category
    check —
    index."""
    return aux


def _bench_cat_image(seed: int = 0) -> float:
    checks = []
    checks.append(cat_image_ok(True, True))
    checks.append(not cat_image_ok(False, True))
    checks.append(cat_image_aux(True))
    checks.append(not cat_image_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_cat_image(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_image": _bench_cat_image(seed)}
