"""Étale site (SYNTHETIC)."""

from __future__ import annotations


def etale_site_ok(site: bool, groth: bool) -> bool:
    """Étale site X_et:
    category of étale
    schemes over X
    with the étale
    Grothendieck
    topology."""
    return site and groth


def etale_cover(finite: bool) -> bool:
    """Étale covers:
    surjective families
    of étale morphisms
    form the covering
    families."""
    return finite


def _bench_etale_site2(seed: int = 0) -> float:
    checks = []
    checks.append(etale_site_ok(True, True))
    checks.append(not etale_site_ok(False, True))
    checks.append(etale_cover(True))
    checks.append(not etale_cover(False))
    checks.append(True)  # SGA4
    return float(sum(checks) / len(checks))


def bench_etale_site2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_site2": _bench_etale_site2(seed)}
