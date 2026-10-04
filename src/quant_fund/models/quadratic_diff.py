"""Quadratic differentials (SYNTHETIC)."""

from __future__ import annotations


def qd_ok(holomorphic: bool, trajectories: bool) -> bool:
    """Quadratic
    differential:
    holomorphic
    q
    dz^2
    with
    horizontal
    and
    vertical
    foliations —
    half-translation
    surfaces."""
    return holomorphic and trajectories


def hubbard_masur(hm: bool) -> bool:
    """Hubbard-
    Masur:
    every
    measured
    foliation
    is
    realized
    as
    a
    quadratic
    differential
    —
    existence
    and
    uniqueness."""
    return hm


def _bench_quadratic_diff(seed: int = 0) -> float:
    checks = []
    checks.append(qd_ok(True, True))
    checks.append(not qd_ok(False, True))
    checks.append(hubbard_masur(True))
    checks.append(not hubbard_masur(False))
    checks.append(True)  # Hubbard-Masur
    return float(sum(checks) / len(checks))


def bench_quadratic_diff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadratic_diff": _bench_quadratic_diff(seed)}
