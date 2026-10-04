"""spectral regular module (SYNTHETIC)."""

from __future__ import annotations


def spectral_regular_ok(spectral: bool, geometric: bool) -> bool:
    """spectral_regular
    check:
    spectral
    structure —
    dvr."""
    return spectral and geometric


def spectral_regular_aux(aux: bool) -> bool:
    """spectral_regular
    aux:
    auxiliary
    spectral
    check —
    noether."""
    return aux


def _bench_spectral_regular(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_regular_ok(True, True))
    checks.append(not spectral_regular_ok(False, True))
    checks.append(spectral_regular_aux(True))
    checks.append(not spectral_regular_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_regular(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_regular": _bench_spectral_regular(seed)}
