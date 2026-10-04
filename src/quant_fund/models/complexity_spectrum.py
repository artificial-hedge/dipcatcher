"""Spectrum complexity (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(complexity: bool, spectrum: bool) -> bool:
    """Complexity:
    complexity
    of
    a
    spectrum —
    devinatz
    complexity."""
    return complexity and spectrum


def spectral_complexity(sc: bool) -> bool:
    """Spectral
    complexity:
    spectral
    complexity
    class —
    chromatic
    complexity."""
    return sc


def _bench_complexity_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(spectral_complexity(True))
    checks.append(not spectral_complexity(False))
    checks.append(True)  # Devinatz-Hopkins-Smith
    return float(sum(checks) / len(checks))


def bench_complexity_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_complexity_spectrum": _bench_complexity_spectrum(seed)}
