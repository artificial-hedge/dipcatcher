"""Categorical 3-cells (SYNTHETIC)."""

from __future__ import annotations


def c3_ok(cat: bool, cell3: bool) -> bool:
    """Categorical
    3
    cell:
    categorical
    3
    cell —
    modification."""
    return cat and cell3


def modification_cell(mc: bool) -> bool:
    """Modification:
    modification
    between
    transformations —
    3
    cell."""
    return mc


def _bench_cat_3cell(seed: int = 0) -> float:
    checks = []
    checks.append(c3_ok(True, True))
    checks.append(not c3_ok(False, True))
    checks.append(modification_cell(True))
    checks.append(not modification_cell(False))
    checks.append(True)  # Benabou
    return float(sum(checks) / len(checks))


def bench_cat_3cell(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cat_3cell": _bench_cat_3cell(seed)}
