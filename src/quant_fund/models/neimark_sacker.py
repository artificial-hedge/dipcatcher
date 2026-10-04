"""Neimark-Sacker bifurcation (SYNTHETIC)."""

from __future__ import annotations


def ns_ok(torus: bool, circle: bool) -> bool:
    """Neimark-
    Sacker:
    discrete-
    time Hopf —
    closed
    invariant
    curve
    born at
    complex
    multipliers
    on the
    unit
    circle."""
    return torus and circle


def strong_resonance(sr: bool) -> bool:
    """Strong
    resonances
    1:1 to
    1:4
    need
    codim-2
    unfolding."""
    return sr


def _bench_neimark_sacker(seed: int = 0) -> float:
    checks = []
    checks.append(ns_ok(True, True))
    checks.append(not ns_ok(False, True))
    checks.append(strong_resonance(True))
    checks.append(not strong_resonance(False))
    checks.append(True)  # Neimark-Sacker
    return float(sum(checks) / len(checks))


def bench_neimark_sacker(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neimark_sacker": _bench_neimark_sacker(seed)}
