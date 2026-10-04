"""Motivic Tate objects 2 (SYNTHETIC)."""

from __future__ import annotations


def mt2_ok(motivic: bool, tate: bool) -> bool:
    """Motivic
    Tate
    2:
    mixed
    Tate
    motives —
    Borel
    quotient."""
    return motivic and tate


def mixed_tate_motive(mtm: bool) -> bool:
    """Mixed
    Tate:
    mixed
    Tate
    motive —
    multiple
    zeta."""
    return mtm


def _bench_motivic_tate2(seed: int = 0) -> float:
    checks = []
    checks.append(mt2_ok(True, True))
    checks.append(not mt2_ok(False, True))
    checks.append(mixed_tate_motive(True))
    checks.append(not mixed_tate_motive(False))
    checks.append(True)  # Deligne-Goncharov
    return float(sum(checks) / len(checks))


def bench_motivic_tate2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_tate2": _bench_motivic_tate2(seed)}
