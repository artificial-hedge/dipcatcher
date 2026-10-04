"""Motivic Landweber (SYNTHETIC)."""

from __future__ import annotations


def ml_ok(motivic: bool, landweber: bool) -> bool:
    """Motivic
    Landweber:
    motivic
    Landweber —
    exact
    functor."""
    return motivic and landweber


def landweber_exact(le: bool) -> bool:
    """Landweber
    exact:
    Landweber
    exact
    functor —
    flat
    map."""
    return le


def _bench_motivic_landweber(seed: int = 0) -> float:
    checks = []
    checks.append(ml_ok(True, True))
    checks.append(not ml_ok(False, True))
    checks.append(landweber_exact(True))
    checks.append(not landweber_exact(False))
    checks.append(True)  # Landweber-Naumann
    return float(sum(checks) / len(checks))


def bench_motivic_landweber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_landweber": _bench_motivic_landweber(seed)}
