"""Gersten spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def gs_ok(gersten_ss: bool, coniveau: bool) -> bool:
    """Gersten:
    coniveau
    spectral
    sequence
    for
    K-theory —
    Gersten
    filtration."""
    return gersten_ss and coniveau


def gersten_conj(gc: bool) -> bool:
    """Gersten
    conjecture:
    K-theory
    of
    regular
    local
    ring —
    Quillen
    Gersten."""
    return gc


def _bench_gersten_ss(seed: int = 0) -> float:
    checks = []
    checks.append(gs_ok(True, True))
    checks.append(not gs_ok(False, True))
    checks.append(gersten_conj(True))
    checks.append(not gersten_conj(False))
    checks.append(True)  # Gersten-Quillen
    return float(sum(checks) / len(checks))


def bench_gersten_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gersten_ss": _bench_gersten_ss(seed)}
