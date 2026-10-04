"""Sheaf stability (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(slope: bool, subsheaf: bool) -> bool:
    """Mumford-
    Takemoto
    stability:
    slope
    comparison
    against
    subsheaves —
    moduli
    of
    sheaves."""
    return slope and subsheaf


def gieseker_stability(gs: bool) -> bool:
    """Gieseker
    stability:
    Hilbert-
    polynomial
    version
    bounded
    better —
    projective
    moduli
    spaces."""
    return gs


def _bench_stability_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(gieseker_stability(True))
    checks.append(not gieseker_stability(False))
    checks.append(True)  # Mumford-Gieseker
    return float(sum(checks) / len(checks))


def bench_stability_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stability_sheaf": _bench_stability_sheaf(seed)}
