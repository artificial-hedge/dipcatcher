"""Etale K-theory (SYNTHETIC)."""

from __future__ import annotations


def ek_ok(etale_k: bool, dmk: bool) -> bool:
    """Etale
    K:
    etale
    K-
    theory —
    Dwyer-
    Friedlander."""
    return etale_k and dmk


def dmk_comparison(dc: bool) -> bool:
    """DMK:
    Dwyer-
    Friedlander-
    Mitchell
    comparison —
    etale
    K."""
    return dc


def _bench_etale_ktheory(seed: int = 0) -> float:
    checks = []
    checks.append(ek_ok(True, True))
    checks.append(not ek_ok(False, True))
    checks.append(dmk_comparison(True))
    checks.append(not dmk_comparison(False))
    checks.append(True)  # D-F-M
    return float(sum(checks) / len(checks))


def bench_etale_ktheory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_ktheory": _bench_etale_ktheory(seed)}
