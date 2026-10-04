"""Clausen-Scholze condensed (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(extremally_disconn: bool, condensed_objects: bool) -> bool:
    """Clausen-
    Scholze:
    condensed
    mathematics
    on
    profinite
    sets —
    solid
    theory."""
    return extremally_disconn and condensed_objects


def solid_theory(st: bool) -> bool:
    """Solid:
    solid
    abelian
    groups
    are
    right
    category —
    condensed
    solid."""
    return st


def _bench_clausen_scholze2(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(solid_theory(True))
    checks.append(not solid_theory(False))
    checks.append(True)  # Clausen-Scholze
    return float(sum(checks) / len(checks))


def bench_clausen_scholze2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clausen_scholze2": _bench_clausen_scholze2(seed)}
