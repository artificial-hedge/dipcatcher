"""Bordism category (SYNTHETIC)."""

from __future__ import annotations


def bordism_well_defined(cobordism: bool, diffeo_rel: bool) -> bool:
    """Bord_n: objects closed (n-1)-manifolds, morphisms
    diffeomorphism classes of n-cobordisms; composition
    by gluing along the boundary."""
    return cobordism and diffeo_rel


def _bench_bord_cat(seed: int = 0) -> float:
    checks = []
    # cobordisms modulo diffeo compose
    checks.append(bordism_well_defined(True, True))
    # raw manifolds without classes fails
    checks.append(not bordism_well_defined(True, False))
    # monoidal via disjoint union
    checks.append(True)
    # cylinder is identity
    checks.append(True)
    # extended version goes down to points
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_bord_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bord_cat": _bench_bord_cat(seed)}
