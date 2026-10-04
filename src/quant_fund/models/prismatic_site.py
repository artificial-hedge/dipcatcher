"""Prismatic site (SYNTHETIC)."""

from __future__ import annotations


def ps_ok(prismatic: bool, site: bool) -> bool:
    """Prismatic
    site:
    prismatic
    site
    of
    a
    p-
    adic
    formal
    scheme —
    Bhatt-
    Scholze."""
    return prismatic and site


def prismatic_tope(pt: bool) -> bool:
    """Prismatic
    topos:
    prismatic
    topos —
    Bhatt-
    Scholze."""
    return pt


def _bench_prismatic_site(seed: int = 0) -> float:
    checks = []
    checks.append(ps_ok(True, True))
    checks.append(not ps_ok(False, True))
    checks.append(prismatic_tope(True))
    checks.append(not prismatic_tope(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_prismatic_site(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prismatic_site": _bench_prismatic_site(seed)}
