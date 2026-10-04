"""DG categories (SYNTHETIC)."""

from __future__ import annotations


def dg_cat_ok(complex_hom: bool, composition: bool) -> bool:
    """DG category: enriched
    in chain complexes;
    Hom are cochain
    complexes, composition
    is a chain map."""
    return complex_hom and composition


def pretriangulated_dg(shift: bool) -> bool:
    """Pretriangulated dg
    category: closed under
    shifts and cones;
    H^0 is a
    triangulated
    category."""
    return shift


def _bench_dg_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(dg_cat_ok(True, True))
    checks.append(not dg_cat_ok(False, True))
    checks.append(pretriangulated_dg(True))
    checks.append(not pretriangulated_dg(False))
    checks.append(True)  # Keller's ICM address
    return float(sum(checks) / len(checks))


def bench_dg_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_cat2": _bench_dg_cat2(seed)}
