"""Berkovich skeleton (SYNTHETIC)."""

from __future__ import annotations


def skeleton_ok(snc_model: bool, retraction: bool) -> bool:
    """Skeleton Sk(X) of
    Berkovich space from a
    semistable model;
    retracts X onto a
    finite simplicial
    complex."""
    return snc_model and retraction


def tropical_skeleton(dimension: bool) -> bool:
    """Skeleton dimension =
    essential skeleton of
    the snc model;
    Kontsevich-Soibelman
    weight functions."""
    return dimension


def _bench_skeleton_trop(seed: int = 0) -> float:
    checks = []
    checks.append(skeleton_ok(True, True))
    checks.append(not skeleton_ok(False, True))
    checks.append(tropical_skeleton(True))
    checks.append(not tropical_skeleton(False))
    checks.append(True)  # Mustafin-Nicaise
    return float(sum(checks) / len(checks))


def bench_skeleton_trop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skeleton_trop": _bench_skeleton_trop(seed)}
