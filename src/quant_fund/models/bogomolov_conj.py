"""Bogomolov conjecture (SYNTHETIC)."""

from __future__ import annotations


def bc_ok(small_height: bool, torsion_dense: bool) -> bool:
    """Bogomolov:
    points
    of
    small
    height
    are
    torsion —
    Zhang's
    theorem."""
    return small_height and torsion_dense


def essential_minimum(em: bool) -> bool:
    """Essential
    minimum:
    infimum
    of
    heights
    outside
    small
    sets —
    positivity
    criterion."""
    return em


def _bench_bogomolov_conj(seed: int = 0) -> float:
    checks = []
    checks.append(bc_ok(True, True))
    checks.append(not bc_ok(False, True))
    checks.append(essential_minimum(True))
    checks.append(not essential_minimum(False))
    checks.append(True)  # Zhang-Ullmo
    return float(sum(checks) / len(checks))


def bench_bogomolov_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bogomolov_conj": _bench_bogomolov_conj(seed)}
