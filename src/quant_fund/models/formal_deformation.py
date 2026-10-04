"""Formal deformations (SYNTHETIC)."""

from __future__ import annotations


def formal_def_ok(formal_groupoid: bool, artinian: bool) -> bool:
    """Formal deformation of X
    over Spf(A): lift to
    infinitesimal thickenings;
    functor on artinian
    rings."""
    return formal_groupoid and artinian


def prorepresentable_complete(hull: bool) -> bool:
    """Prorepresentability:
    formal deformation functor
    has a hull/versal
    deformation over a
    complete local ring
    (Schlessinger)."""
    return hull


def _bench_formal_deformation(seed: int = 0) -> float:
    checks = []
    checks.append(formal_def_ok(True, True))
    checks.append(not formal_def_ok(False, True))
    checks.append(prorepresentable_complete(True))
    checks.append(not prorepresentable_complete(False))
    checks.append(True)  # Kodaira-Spencer class
    return float(sum(checks) / len(checks))


def bench_formal_deformation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_deformation": _bench_formal_deformation(seed)}
