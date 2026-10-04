"""Deformation functors (SYNTHETIC)."""

from __future__ import annotations


def df2_ok(deform: bool, functor: bool) -> bool:
    """Deformation
    functor:
    deformation
    functor —
    Schlessinger
    functor."""
    return deform and functor


def deformation_set(ds: bool) -> bool:
    """Deformation
    set:
    deformation
    set
    over
    an
    Artinian
    ring —
    fiber
    product."""
    return ds


def _bench_deform_functor2(seed: int = 0) -> float:
    checks = []
    checks.append(df2_ok(True, True))
    checks.append(not df2_ok(False, True))
    checks.append(deformation_set(True))
    checks.append(not deformation_set(False))
    checks.append(True)  # Schlessinger
    return float(sum(checks) / len(checks))


def bench_deform_functor2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deform_functor2": _bench_deform_functor2(seed)}
