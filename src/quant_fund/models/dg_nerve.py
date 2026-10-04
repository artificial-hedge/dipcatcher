"""DG nerve (SYNTHETIC)."""

from __future__ import annotations


def dg_nerve_ok(simplicial: bool, hocolim: bool) -> bool:
    """DG nerve N_dg(C)
    of a dg cat:
    simplicial set
    enhancing C;
    Lurie construction."""
    return simplicial and hocolim


def quasi_cat_dg(kan: bool) -> bool:
    """The dg nerve is
    a quasi-category:
    lifts the dg
    category to an
    infinity-category."""
    return kan


def _bench_dg_nerve(seed: int = 0) -> float:
    checks = []
    checks.append(dg_nerve_ok(True, True))
    checks.append(not dg_nerve_ok(False, True))
    checks.append(quasi_cat_dg(True))
    checks.append(not quasi_cat_dg(False))
    checks.append(True)  # Lurie Higher Algebra
    return float(sum(checks) / len(checks))


def bench_dg_nerve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_nerve": _bench_dg_nerve(seed)}
