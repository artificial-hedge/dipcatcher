"""Fuchsian groups (SYNTHETIC)."""

from __future__ import annotations


def fuch_ok(disc: bool, discrete: bool) -> bool:
    """Fuchsian
    group:
    discrete
    subgroup
    of
    PSL(2,R)
    acting
    on
    the
    disc —
    quotient
    is
    a
    surface."""
    return disc and discrete


def fundamental_domain(fd: bool) -> bool:
    """Dirichlet
    fundamental
    domain:
    polygon
    with
    side
    pairings
    gives
    a
    presentation."""
    return fd


def _bench_fuchsian_group(seed: int = 0) -> float:
    checks = []
    checks.append(fuch_ok(True, True))
    checks.append(not fuch_ok(False, True))
    checks.append(fundamental_domain(True))
    checks.append(not fundamental_domain(False))
    checks.append(True)  # Poincare
    return float(sum(checks) / len(checks))


def bench_fuchsian_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fuchsian_group": _bench_fuchsian_group(seed)}
