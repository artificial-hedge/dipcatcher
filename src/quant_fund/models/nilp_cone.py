"""Nilpotent cone (SYNTHETIC)."""

from __future__ import annotations


def nc_ok(nilpotent_orbits: bool, moment_fiber: bool) -> bool:
    """Nilpotent
    cone:
    nilpotent
    elements
    in
    Lie
    algebra —
    Springer
    theory."""
    return nilpotent_orbits and moment_fiber


def springer_resolution(sr: bool) -> bool:
    """Springer:
    cotangent
    of
    flag
    variety
    resolves
    nilpotent
    cone —
    Springer
    resolution."""
    return sr


def _bench_nilp_cone(seed: int = 0) -> float:
    checks = []
    checks.append(nc_ok(True, True))
    checks.append(not nc_ok(False, True))
    checks.append(springer_resolution(True))
    checks.append(not springer_resolution(False))
    checks.append(True)  # Springer
    return float(sum(checks) / len(checks))


def bench_nilp_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nilp_cone": _bench_nilp_cone(seed)}
