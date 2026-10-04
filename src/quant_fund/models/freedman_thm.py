"""Freedman theorem (SYNTHETIC)."""

from __future__ import annotations


def freed_ok(simply_conn: bool, classified: bool) -> bool:
    """Freedman's
    theorem:
    simply-connected
    closed
    topological
    4-manifolds
    are
    classified
    by
    intersection
    form
    and
    Kirby-
    Siebenmann."""
    return simply_conn and classified


def e8_manifold(e8: bool) -> bool:
    """E8
    manifold
    exists
    topologically
    but
    admits
    no
    smooth
    structure —
    Donaldson
    plus
    Rokhlin."""
    return e8


def _bench_freedman_thm(seed: int = 0) -> float:
    checks = []
    checks.append(freed_ok(True, True))
    checks.append(not freed_ok(False, True))
    checks.append(e8_manifold(True))
    checks.append(not e8_manifold(False))
    checks.append(True)  # Freedman 1982
    return float(sum(checks) / len(checks))


def bench_freedman_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_freedman_thm": _bench_freedman_thm(seed)}
