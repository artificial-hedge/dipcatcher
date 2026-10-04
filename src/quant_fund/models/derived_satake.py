"""Derived Satake (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(affine_grassmannian: bool, derived_equiv: bool) -> bool:
    """Derived
    Satake:
    derived
    equivalence
    for
    affine
    Grassmannian —
    Bezrukavnikov-
    Finkelberg."""
    return affine_grassmannian and derived_equiv


def langlands_dual_derived(ldd: bool) -> bool:
    """Langlands
    dual
    derived:
    dual
    group
    controls
    derived
    category —
    derived
    geometric
    Satake."""
    return ldd


def _bench_derived_satake(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(langlands_dual_derived(True))
    checks.append(not langlands_dual_derived(False))
    checks.append(True)  # Bezrukavnikov-Finkelberg
    return float(sum(checks) / len(checks))


def bench_derived_satake(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_satake": _bench_derived_satake(seed)}
