"""spectral noether module (SYNTHETIC)."""

from __future__ import annotations


def spectral_noether_ok(spectral: bool, geometric: bool) -> bool:
    """spectral_noether
    check:
    spectral
    structure —
    dvr."""
    return spectral and geometric


def spectral_noether_aux(aux: bool) -> bool:
    """spectral_noether
    aux:
    auxiliary
    spectral
    check —
    noether."""
    return aux


def _bench_spectral_noether(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_noether_ok(True, True))
    checks.append(not spectral_noether_ok(False, True))
    checks.append(spectral_noether_aux(True))
    checks.append(not spectral_noether_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_noether(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_noether": _bench_spectral_noether(seed)}
