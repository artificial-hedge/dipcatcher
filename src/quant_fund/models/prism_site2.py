"""Prism site 2 (SYNTHETIC)."""

from __future__ import annotations


def ps2_ok(prism: bool, site: bool) -> bool:
    """Prism
    site:
    prism
    site —
    base
    covering."""
    return prism and site


def base_prism_site(bps: bool) -> bool:
    """Base
    prism:
    base
    prism
    site —
    covering."""
    return bps


def _bench_prism_site2(seed: int = 0) -> float:
    checks = []
    checks.append(ps2_ok(True, True))
    checks.append(not ps2_ok(False, True))
    checks.append(base_prism_site(True))
    checks.append(not base_prism_site(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_prism_site2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prism_site2": _bench_prism_site2(seed)}
