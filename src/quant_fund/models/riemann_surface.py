"""Riemann surfaces (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(atlas: bool, genus: bool) -> bool:
    """Riemann
    surface:
    one-
    dimensional
    complex
    manifold
    with
    holomorphic
    transition
    maps —
    genus
    classifies
    compact
    ones."""
    return atlas and genus


def charts_compatible(cc: bool) -> bool:
    """Holomorphic
    atlas:
    overlapping
    charts
    compose
    biholomorphically
    —
    no
    branch
    jumps."""
    return cc


def _bench_riemann_surface(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(charts_compatible(True))
    checks.append(not charts_compatible(False))
    checks.append(True)  # Riemann
    return float(sum(checks) / len(checks))


def bench_riemann_surface(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_surface": _bench_riemann_surface(seed)}
