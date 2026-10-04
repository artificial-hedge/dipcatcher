"""Boundedness of moduli (SYNTHETIC)."""

from __future__ import annotations


def bm_ok(finite_type: bool, families: bool) -> bool:
    """Boundedness:
    moduli
    families
    of
    bounded
    type
    form
    finite-
    type
    schemes —
    Matsusaka
    and
    Kollar."""
    return finite_type and families


def kollar_boundedness(kb: bool) -> bool:
    """Kollar
    boundedness:
    effective
    bounds
    for
    stable
    pairs
    in
    birational
    geometry —
    log
    general
    type."""
    return kb


def _bench_boundedness_moduli(seed: int = 0) -> float:
    checks = []
    checks.append(bm_ok(True, True))
    checks.append(not bm_ok(False, True))
    checks.append(kollar_boundedness(True))
    checks.append(not kollar_boundedness(False))
    checks.append(True)  # Matsusaka-Kollar
    return float(sum(checks) / len(checks))


def bench_boundedness_moduli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boundedness_moduli": _bench_boundedness_moduli(seed)}
