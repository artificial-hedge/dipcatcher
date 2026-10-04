"""Rim deformation theory (SYNTHETIC)."""

from __future__ import annotations


def rd_ok(rim: bool, deform: bool) -> bool:
    """Rim
    deformation:
    Rim
    deformation
    theory —
    Rim
    Schlessinger."""
    return rim and deform


def rim_condition(rc: bool) -> bool:
    """Rim
    condition:
    Rim
    condition
    for
    deformation
    functors —
    Rim
    H
    condition."""
    return rc


def _bench_rim_deform(seed: int = 0) -> float:
    checks = []
    checks.append(rd_ok(True, True))
    checks.append(not rd_ok(False, True))
    checks.append(rim_condition(True))
    checks.append(not rim_condition(False))
    checks.append(True)  # Rim
    return float(sum(checks) / len(checks))


def bench_rim_deform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rim_deform": _bench_rim_deform(seed)}
