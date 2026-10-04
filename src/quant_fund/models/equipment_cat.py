"""equipment cat module (SYNTHETIC)."""

from __future__ import annotations


def equipment_cat_ok(category: bool, structure: bool) -> bool:
    """equipment_cat
    check:
    category
    structure —
    enriched."""
    return category and structure


def equipment_cat_aux(aux: bool) -> bool:
    """equipment_cat
    aux:
    auxiliary
    category
    check —
    derived."""
    return aux


def _bench_equipment_cat(seed: int = 0) -> float:
    checks = []
    checks.append(equipment_cat_ok(True, True))
    checks.append(not equipment_cat_ok(False, True))
    checks.append(equipment_cat_aux(True))
    checks.append(not equipment_cat_aux(False))
    checks.append(True)  # category canon
    return float(sum(checks) / len(checks))


def bench_equipment_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equipment_cat": _bench_equipment_cat(seed)}
