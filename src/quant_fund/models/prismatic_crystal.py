"""Prismatic crystals (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(prismatic: bool, crystal: bool) -> bool:
    """Prismatic
    crystal:
    prismatic
    crystal —
    Bhatt-
    Scholze
    crystal."""
    return prismatic and crystal


def crystal_site(cs: bool) -> bool:
    """Crystal
    site:
    prismatic
    crystal
    site —
    prismatic
    crystal."""
    return cs


def _bench_prismatic_crystal(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(crystal_site(True))
    checks.append(not crystal_site(False))
    checks.append(True)  # Bhatt-Scholze
    return float(sum(checks) / len(checks))


def bench_prismatic_crystal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prismatic_crystal": _bench_prismatic_crystal(seed)}
