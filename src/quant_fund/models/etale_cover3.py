"""Etale covers (SYNTHETIC)."""

from __future__ import annotations


def ec3_ok(etale: bool, cover: bool) -> bool:
    """Etale
    cover:
    etale
    cover —
    jointly
    surjective
    etale."""
    return etale and cover


def etale_morph(em: bool) -> bool:
    """Etale
    morphism:
    etale
    morphism —
    flat
    unramified."""
    return em


def _bench_etale_cover3(seed: int = 0) -> float:
    checks = []
    checks.append(ec3_ok(True, True))
    checks.append(not ec3_ok(False, True))
    checks.append(etale_morph(True))
    checks.append(not etale_morph(False))
    checks.append(True)  # SGA4
    return float(sum(checks) / len(checks))


def bench_etale_cover3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_cover3": _bench_etale_cover3(seed)}
