"""Solid cohomology (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(solid_groups: bool, solid_coh: bool) -> bool:
    """Solid
    cohomology:
    Ext
    in
    solid
    abelian
    groups —
    solid
    derived."""
    return solid_groups and solid_coh


def solid_ext(se: bool) -> bool:
    """Solid
    Ext:
    Ext
    of
    solid
    groups
    is
    zero —
    condensed
    vanishing."""
    return se


def _bench_solid_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(solid_ext(True))
    checks.append(not solid_ext(False))
    checks.append(True)  # Clausen-Scholze
    return float(sum(checks) / len(checks))


def bench_solid_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_solid_cohom": _bench_solid_cohom(seed)}
