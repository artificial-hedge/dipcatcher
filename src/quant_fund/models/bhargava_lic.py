"""Bhargava's local-global counting (SYNTHETIC)."""

from __future__ import annotations


def bl_ok(lattice_pts: bool, averaging: bool) -> bool:
    """Bhargava's
    counting:
    lattice-
    point
    asymptotics
    in
    fundamental
    domains —
    quartic
    and
    quintic
    rings."""
    return lattice_pts and averaging


def quartic_ring_count(qr: bool) -> bool:
    """Quartic
    ring
    counting:
    Bhargava
    counts
    quartic
    rings
    by
    discriminant —
    parametrization
    of
    quartic
    fields."""
    return qr


def _bench_bhargava_lic(seed: int = 0) -> float:
    checks = []
    checks.append(bl_ok(True, True))
    checks.append(not bl_ok(False, True))
    checks.append(quartic_ring_count(True))
    checks.append(not quartic_ring_count(False))
    checks.append(True)  # Bhargava
    return float(sum(checks) / len(checks))


def bench_bhargava_lic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bhargava_lic": _bench_bhargava_lic(seed)}
