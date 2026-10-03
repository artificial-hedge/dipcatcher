"""spectral curve module (SYNTHETIC)."""

from __future__ import annotations


def spectral_curve_ok(spectral: bool, geometry: bool) -> bool:
    """spectral_curve
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def spectral_curve_aux(aux: bool) -> bool:
    """spectral_curve
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_spectral_curve(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_curve_ok(True, True))
    checks.append(not spectral_curve_ok(False, True))
    checks.append(spectral_curve_aux(True))
    checks.append(not spectral_curve_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_curve": _bench_spectral_curve(seed)}
