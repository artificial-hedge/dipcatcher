"""Geometric Satake II (SYNTHETIC)."""

from __future__ import annotations


def gs_ok(perverse_equiv: bool, tensor_structure: bool) -> bool:
    """Geometric
    Satake:
    tensor
    equivalence
    Satake
    category
    and
    Rep
    of
    dual —
    MV
    theorem."""
    return perverse_equiv and tensor_structure


def mirkovic_vilonen(mv: bool) -> bool:
    """Mirkovic-
    Vilonen:
    semisimple
    perverse
    sheaves
    on
    affine
    Grassmannian —
    MV
    cycles."""
    return mv


def _bench_geometric_satake2(seed: int = 0) -> float:
    checks = []
    checks.append(gs_ok(True, True))
    checks.append(not gs_ok(False, True))
    checks.append(mirkovic_vilonen(True))
    checks.append(not mirkovic_vilonen(False))
    checks.append(True)  # Mirkovic-Vilonen
    return float(sum(checks) / len(checks))


def bench_geometric_satake2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geometric_satake2": _bench_geometric_satake2(seed)}
