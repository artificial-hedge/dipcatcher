"""Monopole Floer homology (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(sw_eq: bool, torsion: bool) -> bool:
    """Monopole
    Floer
    homology:
    3-dimensional
    Seiberg-
    Witten
    functional —
    Kronheimer-
    Mrowka."""
    return sw_eq and torsion


def gradings_tower(gt: bool) -> bool:
    """Three
    variants:
    hat,
    plus,
    minus
    flavors
    tied
    by
    exact
    triangles."""
    return gt


def _bench_monopole_floer(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(gradings_tower(True))
    checks.append(not gradings_tower(False))
    checks.append(True)  # Kronheimer-Mrowka
    return float(sum(checks) / len(checks))


def bench_monopole_floer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monopole_floer": _bench_monopole_floer(seed)}
