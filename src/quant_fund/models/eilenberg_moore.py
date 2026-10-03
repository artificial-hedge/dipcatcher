"""Eilenberg-Moore spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def em_ok(pullback_fiber: bool, tor_spectral: bool) -> bool:
    """Eilenberg-
    Moore:
    Tor
    spectral
    sequence
    for
    pullback
    cohomology —
    EMSS."""
    return pullback_fiber and tor_spectral


def em_convergence(emc: bool) -> bool:
    """EM
    convergence:
    Tor
    of
    cohomology
    converges
    to
    cohomology
    of
    pullback —
    Eilenberg-
    Moore."""
    return emc


def _bench_eilenberg_moore(seed: int = 0) -> float:
    checks = []
    checks.append(em_ok(True, True))
    checks.append(not em_ok(False, True))
    checks.append(em_convergence(True))
    checks.append(not em_convergence(False))
    checks.append(True)  # Eilenberg-Moore
    return float(sum(checks) / len(checks))


def bench_eilenberg_moore(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eilenberg_moore": _bench_eilenberg_moore(seed)}
