"""Spectral groups (SYNTHETIC)."""

from __future__ import annotations


def sg_ok(spectral: bool, group: bool) -> bool:
    """Spectral
    group:
    spectral
    group —
    group
    E
    ring."""
    return spectral and group


def group_e_ring(ge: bool) -> bool:
    """Group
    E
    ring:
    group
    E-infinity
    ring —
    units."""
    return ge


def _bench_spectral_group(seed: int = 0) -> float:
    checks = []
    checks.append(sg_ok(True, True))
    checks.append(not sg_ok(False, True))
    checks.append(group_e_ring(True))
    checks.append(not group_e_ring(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_spectral_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_group": _bench_spectral_group(seed)}
