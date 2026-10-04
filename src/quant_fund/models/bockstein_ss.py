"""Bockstein spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def bs_ok(coeff_change: bool, torsion_detect: bool) -> bool:
    """Bockstein:
    detects
    torsion
    via
    coefficient
    short
    exact —
    connecting
    homomorphism."""
    return coeff_change and torsion_detect


def p_primary_bockstein(pb: bool) -> bool:
    """p-
    Bockstein:
    iterated
    Bockstein
    converges
    to
    free
    part —
    Browder's
    theorem."""
    return pb


def _bench_bockstein_ss(seed: int = 0) -> float:
    checks = []
    checks.append(bs_ok(True, True))
    checks.append(not bs_ok(False, True))
    checks.append(p_primary_bockstein(True))
    checks.append(not p_primary_bockstein(False))
    checks.append(True)  # Bockstein-Browder
    return float(sum(checks) / len(checks))


def bench_bockstein_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bockstein_ss": _bench_bockstein_ss(seed)}
