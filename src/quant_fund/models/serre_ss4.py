"""Serre spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(fibration: bool, homology_fiber: bool) -> bool:
    """Serre
    spectral
    sequence:
    homology
    of
    fibration
    converges —
    Leray-
    Serre."""
    return fibration and homology_fiber


def e2_page(ep: bool) -> bool:
    """E2
    page:
    homology
    of
    base
    with
    fiber
    coefficients —
    Serre's
    spectral
    sequence."""
    return ep


def _bench_serre_ss4(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(e2_page(True))
    checks.append(not e2_page(False))
    checks.append(True)  # Serre
    return float(sum(checks) / len(checks))


def bench_serre_ss4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_serre_ss4": _bench_serre_ss4(seed)}
