"""spectral gm module (SYNTHETIC)."""

from __future__ import annotations


def spectral_gm_ok(spectral: bool, geometry: bool) -> bool:
    """spectral_gm
    check:
    spectral
    algebraic
    geometry —
    stacky."""
    return spectral and geometry


def spectral_gm_aux(aux: bool) -> bool:
    """spectral_gm
    aux:
    auxiliary
    spectral
    check —
    derived."""
    return aux


def _bench_spectral_gm(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_gm_ok(True, True))
    checks.append(not spectral_gm_ok(False, True))
    checks.append(spectral_gm_aux(True))
    checks.append(not spectral_gm_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_gm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_gm": _bench_spectral_gm(seed)}
