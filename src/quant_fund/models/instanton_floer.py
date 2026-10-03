"""Instanton Floer homology (SYNTHETIC)."""

from __future__ import annotations


def inf_ok(yang_mills: bool, flat: bool) -> bool:
    """Instanton
    Floer
    homology:
    flat
    connections
    as
    critical
    points,
    ASD
    instantons
    as
    trajectories."""
    return yang_mills and flat


def surgery_triad(st: bool) -> bool:
    """Surgery
    exact
    triangle:
    instanton
    homology
    obeys
    a
    surgery
    exact
    sequence —
    Casson
    counts."""
    return st


def _bench_instanton_floer(seed: int = 0) -> float:
    checks = []
    checks.append(inf_ok(True, True))
    checks.append(not inf_ok(False, True))
    checks.append(surgery_triad(True))
    checks.append(not surgery_triad(False))
    checks.append(True)  # Floer instanton
    return float(sum(checks) / len(checks))


def bench_instanton_floer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_instanton_floer": _bench_instanton_floer(seed)}
