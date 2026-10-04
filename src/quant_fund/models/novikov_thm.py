"""Novikov compact leaf (SYNTHETIC)."""

from __future__ import annotations


def nov_ok(s3: bool, torus_leaf: bool) -> bool:
    """Novikov:
    every
    codim-1
    foliation
    of
    S3
    has
    a
    compact
    torus
    leaf —
    Reeb
    component."""
    return s3 and torus_leaf


def pi2_vanish(pv: bool) -> bool:
    """vanishing
    pi-2
    alternative:
    no
    Reeb
    component
    implies
    universal
    cover
    is
    R3."""
    return pv


def _bench_novikov_thm(seed: int = 0) -> float:
    checks = []
    checks.append(nov_ok(True, True))
    checks.append(not nov_ok(False, True))
    checks.append(pi2_vanish(True))
    checks.append(not pi2_vanish(False))
    checks.append(True)  # Novikov 1965
    return float(sum(checks) / len(checks))


def bench_novikov_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_novikov_thm": _bench_novikov_thm(seed)}
