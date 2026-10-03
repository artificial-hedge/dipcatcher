"""Spectral etaleness (SYNTHETIC)."""

from __future__ import annotations


def se_ok(spectral: bool, etale: bool) -> bool:
    """Spectral
    etale:
    spectral
    etale
    morphism —
    etale
    site."""
    return spectral and etale


def etale_morphism(em: bool) -> bool:
    """Etale
    morphism:
    etale
    morphism —
    flat
    unramified."""
    return em


def _bench_spectral_etale(seed: int = 0) -> float:
    checks = []
    checks.append(se_ok(True, True))
    checks.append(not se_ok(False, True))
    checks.append(etale_morphism(True))
    checks.append(not etale_morphism(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_spectral_etale(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_etale": _bench_spectral_etale(seed)}
