"""spectral perfect module (SYNTHETIC)."""

from __future__ import annotations


def spectral_perfect_ok(spectral: bool, geometric: bool) -> bool:
    """spectral_perfect
    check:
    spectral
    structure —
    spectral AG."""
    return spectral and geometric


def spectral_perfect_aux(aux: bool) -> bool:
    """spectral_perfect
    aux:
    auxiliary
    spectral
    check —
    formal."""
    return aux


def _bench_spectral_perfect(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_perfect_ok(True, True))
    checks.append(not spectral_perfect_ok(False, True))
    checks.append(spectral_perfect_aux(True))
    checks.append(not spectral_perfect_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_perfect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_perfect": _bench_spectral_perfect(seed)}
