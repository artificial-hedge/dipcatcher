"""Cone theorem (SYNTHETIC)."""

from __future__ import annotations


def ct_ok(cone: bool, theorem: bool) -> bool:
    """Cone
    theorem:
    cone
    theorem —
    Mori
    cone."""
    return cone and theorem


def ne_cone(ne: bool) -> bool:
    """NE
    cone:
    cone
    of
    curves —
    extremal
    rays."""
    return ne


def _bench_cone_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(ct_ok(True, True))
    checks.append(not ct_ok(False, True))
    checks.append(ne_cone(True))
    checks.append(not ne_cone(False))
    checks.append(True)  # Mori
    return float(sum(checks) / len(checks))


def bench_cone_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cone_theorem": _bench_cone_theorem(seed)}
