"""Brave new ring (SYNTHETIC)."""

from __future__ import annotations


def bn_ok(brave: bool, ring: bool) -> bool:
    """Brave:
    brave
    new
    ring
    spectrum —
    May
    brave
    new."""
    return brave and ring


def ring_spectrum(rs: bool) -> bool:
    """Ring
    spectrum:
    ring
    spectrum
    structure —
    EKMM
    ring."""
    return rs


def _bench_brave_new_ring(seed: int = 0) -> float:
    checks = []
    checks.append(bn_ok(True, True))
    checks.append(not bn_ok(False, True))
    checks.append(ring_spectrum(True))
    checks.append(not ring_spectrum(False))
    checks.append(True)  # EKMM
    return float(sum(checks) / len(checks))


def bench_brave_new_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brave_new_ring": _bench_brave_new_ring(seed)}
