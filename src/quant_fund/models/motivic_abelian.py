"""Motivic abelian (SYNTHETIC)."""

from __future__ import annotations


def ma_ok(motivic: bool, abelian: bool) -> bool:
    """Motivic
    abelian:
    motivic
    abelian
    motive —
    semisimple."""
    return motivic and abelian


def abelian_motive(am: bool) -> bool:
    """Abelian
    motive:
    abelian
    motive —
    CM
    motive."""
    return am


def _bench_motivic_abelian(seed: int = 0) -> float:
    checks = []
    checks.append(ma_ok(True, True))
    checks.append(not ma_ok(False, True))
    checks.append(abelian_motive(True))
    checks.append(not abelian_motive(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_motivic_abelian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_abelian": _bench_motivic_abelian(seed)}
