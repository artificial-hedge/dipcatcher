"""Motivic Steenrod (SYNTHETIC)."""

from __future__ import annotations


def ms_ok2(motivic: bool, steenrod: bool) -> bool:
    """Motivic:
    motivic
    Steenrod
    operations —
    Voevodsky
    Sq."""
    return motivic and steenrod


def voev_squares(vs: bool) -> bool:
    """Voevodsky
    squares:
    Voevodsky
    motivic
    Sq
    operations —
    motivic
    Steenrod."""
    return vs


def _bench_motivic_steenrod(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok2(True, True))
    checks.append(not ms_ok2(False, True))
    checks.append(voev_squares(True))
    checks.append(not voev_squares(False))
    checks.append(True)  # Voevodsky
    return float(sum(checks) / len(checks))


def bench_motivic_steenrod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_steenrod": _bench_motivic_steenrod(seed)}
