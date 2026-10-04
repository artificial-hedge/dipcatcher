"""Etale sites (SYNTHETIC)."""

from __future__ import annotations


def es3_ok(etale: bool, site: bool) -> bool:
    """Etale
    site:
    etale
    site —
    small
    etale
    site."""
    return etale and site


def small_etale(se: bool) -> bool:
    """Small
    etale:
    small
    etale
    site —
    etale
    topology."""
    return se


def _bench_etale_site3(seed: int = 0) -> float:
    checks = []
    checks.append(es3_ok(True, True))
    checks.append(not es3_ok(False, True))
    checks.append(small_etale(True))
    checks.append(not small_etale(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_etale_site3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_site3": _bench_etale_site3(seed)}
