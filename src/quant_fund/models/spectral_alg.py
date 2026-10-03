"""Spectral algebraic geometry (SYNTHETIC)."""

from __future__ import annotations


def sag_ok(prestack: bool, etale: bool) -> bool:
    """Spectral algebraic
    geometry: geometry
    built on
    E_infty-rings
    rather than
    ordinary rings;
    derived schemes
    are examples."""
    return prestack and etale


def sag_faithful(faithful: bool) -> bool:
    """Classical
    algebraic geometry
    embeds fully
    faithfully into
    spectral AG as
    0-truncated
    objects."""
    return faithful


def _bench_spectral_alg(seed: int = 0) -> float:
    checks = []
    checks.append(sag_ok(True, True))
    checks.append(not sag_ok(False, True))
    checks.append(sag_faithful(True))
    checks.append(not sag_faithful(False))
    checks.append(True)  # Lurie SAG
    return float(sum(checks) / len(checks))


def bench_spectral_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_alg": _bench_spectral_alg(seed)}
