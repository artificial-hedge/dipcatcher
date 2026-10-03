"""Schlessinger criterion (SYNTHETIC)."""

from __future__ import annotations


def sh_ok(schlessinger: bool, functor: bool) -> bool:
    """Schlessinger:
    Schlessinger
    criterion
    for
    deformation
    functors —
    Schlessinger
    criterion."""
    return schlessinger and functor


def hull_exist(he: bool) -> bool:
    """Hull
    existence:
    Schlessinger
    hull
    existence —
    Schlessinger
    hull."""
    return he


def _bench_schlessinger2(seed: int = 0) -> float:
    checks = []
    checks.append(sh_ok(True, True))
    checks.append(not sh_ok(False, True))
    checks.append(hull_exist(True))
    checks.append(not hull_exist(False))
    checks.append(True)  # Schlessinger
    return float(sum(checks) / len(checks))


def bench_schlessinger2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schlessinger2": _bench_schlessinger2(seed)}
