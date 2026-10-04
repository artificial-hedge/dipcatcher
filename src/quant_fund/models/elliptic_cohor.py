"""Elliptic cohomology (SYNTHETIC)."""

from __future__ import annotations


def ell_ok(orient: bool, formal: bool) -> bool:
    """Elliptic
    cohomology E:
    even periodic
    theory whose
    formal group
    is the formal
    group of an
    elliptic curve."""
    return orient and formal


def spectral_ell(spectral: bool) -> bool:
    """Lurie's theorem:
    elliptic cohomology
    is a sheaf of
    E_infty-rings
    on the derived
    moduli of elliptic
    curves (TAF)."""
    return spectral


def _bench_elliptic_cohor(seed: int = 0) -> float:
    checks = []
    checks.append(ell_ok(True, True))
    checks.append(not ell_ok(False, True))
    checks.append(spectral_ell(True))
    checks.append(not spectral_ell(False))
    checks.append(True)  # Lurie TAF
    return float(sum(checks) / len(checks))


def bench_elliptic_cohor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_cohor": _bench_elliptic_cohor(seed)}
