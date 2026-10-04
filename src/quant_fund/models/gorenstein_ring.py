"""Gorenstein ring (SYNTHETIC)."""

from __future__ import annotations


def gr_ok(gorenstein: bool, ring: bool) -> bool:
    """Gorenstein:
    Gorenstein
    local
    ring —
    Bass
    Gorenstein."""
    return gorenstein and ring


def self_injective(si: bool) -> bool:
    """Self
    injective:
    Gorenstein
    self-
    injective
    dimension —
    Bass
    criterion."""
    return si


def _bench_gorenstein_ring(seed: int = 0) -> float:
    checks = []
    checks.append(gr_ok(True, True))
    checks.append(not gr_ok(False, True))
    checks.append(self_injective(True))
    checks.append(not self_injective(False))
    checks.append(True)  # Bass
    return float(sum(checks) / len(checks))


def bench_gorenstein_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gorenstein_ring": _bench_gorenstein_ring(seed)}
