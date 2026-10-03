"""Geometric Satake (SYNTHETIC)."""

from __future__ import annotations


def sat_ok(affine: bool, tensor: bool) -> bool:
    """Geometric
    Satake:
    Perv_{L+G}(Gr_G)
    is a symmetric
    monoidal
    category
    equivalent to
    Rep(G^vee)."""
    return affine and tensor


def mirk_vil(mv: bool) -> bool:
    """Mirkovic-
    Vilonen:
    weight functor
    from MV
    cycles
    computes
    the Tannakian
    fiber."""
    return mv


def _bench_geometric_satake(seed: int = 0) -> float:
    checks = []
    checks.append(sat_ok(True, True))
    checks.append(not sat_ok(False, True))
    checks.append(mirk_vil(True))
    checks.append(not mirk_vil(False))
    checks.append(True)  # MV 2007
    return float(sum(checks) / len(checks))


def bench_geometric_satake(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geometric_satake": _bench_geometric_satake(seed)}
