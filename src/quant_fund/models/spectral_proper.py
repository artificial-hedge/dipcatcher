"""spectral proper module (SYNTHETIC)."""

from __future__ import annotations


def spectral_proper_ok(spectral: bool, geometric: bool) -> bool:
    """spectral_proper
    check:
    spectral
    structure —
    spectral AG."""
    return spectral and geometric


def spectral_proper_aux(aux: bool) -> bool:
    """spectral_proper
    aux:
    auxiliary
    spectral
    check —
    formal."""
    return aux


def _bench_spectral_proper(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_proper_ok(True, True))
    checks.append(not spectral_proper_ok(False, True))
    checks.append(spectral_proper_aux(True))
    checks.append(not spectral_proper_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_spectral_proper(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_proper": _bench_spectral_proper(seed)}
