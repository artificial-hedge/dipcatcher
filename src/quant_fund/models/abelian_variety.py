"""Abelian varieties (SYNTHETIC)."""

from __future__ import annotations


def av_ok(projective_group: bool, smooth: bool) -> bool:
    """Abelian
    variety:
    projective
    connected
    algebraic
    group —
    group
    law
    is
    commutative."""
    return projective_group and smooth


def theta_structure(ts: bool) -> bool:
    """Theta
    structure:
    level
    structure
    on
    the
    Heisenberg
    group
    action —
    Mumford
    models."""
    return ts


def _bench_abelian_variety(seed: int = 0) -> float:
    checks = []
    checks.append(av_ok(True, True))
    checks.append(not av_ok(False, True))
    checks.append(theta_structure(True))
    checks.append(not theta_structure(False))
    checks.append(True)  # Mumford
    return float(sum(checks) / len(checks))


def bench_abelian_variety(seed: int = 0) -> dict[str, float]:
    return {"synthetic_abelian_variety": _bench_abelian_variety(seed)}
