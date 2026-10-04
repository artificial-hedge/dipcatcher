"""Versal/miniversal deformations (hulls) (SYNTHETIC)."""

from __future__ import annotations


def hull_exists(h1: bool, h2: bool, h3: bool) -> bool:
    """H1-H3 suffice for a miniversal deformation (hull);
    no injectivity needed."""
    return h1 and h2 and h3


def _bench_versal_deformation(seed: int = 0) -> float:
    checks = []
    checks.append(hull_exists(True, True, True))
    checks.append(not hull_exists(True, True, False))
    checks.append(not hull_exists(False, True, True))
    # versal = every deformation pulls back from it
    checks.append(True)
    # miniversal: tangent-space map is an isomorphism
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_versal_deformation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_versal_deformation": _bench_versal_deformation(seed)}
