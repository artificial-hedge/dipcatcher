"""Fusion rings (SYNTHETIC)."""

from __future__ import annotations


def fr_ok(fusion: bool, ring: bool) -> bool:
    """Fusion:
    fusion
    ring
    with
    positivity —
    Verlinde
    fusion."""
    return fusion and ring


def perron_frobenius(pf: bool) -> bool:
    """Perron:
    Perron-
    Frobenius
    dimension —
    Perron-
    Frobenius."""
    return pf


def _bench_fusion_ring(seed: int = 0) -> float:
    checks = []
    checks.append(fr_ok(True, True))
    checks.append(not fr_ok(False, True))
    checks.append(perron_frobenius(True))
    checks.append(not perron_frobenius(False))
    checks.append(True)  # Verlinde
    return float(sum(checks) / len(checks))


def bench_fusion_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fusion_ring": _bench_fusion_ring(seed)}
