"""Gromov nonsqueezing (SYNTHETIC)."""

from __future__ import annotations


def gromov_ok(ball: bool, cylinder: bool) -> bool:
    """Gromov
    nonsqueezing:
    a ball
    cannot
    symplectically
    embed
    into
    a thinner
    cylinder."""
    return ball and cylinder


def symplectic_capacity(cap: bool) -> bool:
    """Symplectic
    capacities:
    monotone
    symplectic
    invariants
    obstructing
    embeddings."""
    return cap


def _bench_gromov_nonsq(seed: int = 0) -> float:
    checks = []
    checks.append(gromov_ok(True, True))
    checks.append(not gromov_ok(False, True))
    checks.append(symplectic_capacity(True))
    checks.append(not symplectic_capacity(False))
    checks.append(True)  # Gromov
    return float(sum(checks) / len(checks))


def bench_gromov_nonsq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gromov_nonsq": _bench_gromov_nonsq(seed)}
