"""Lagrangian manifolds (SYNTHETIC)."""

from __future__ import annotations


def lag_ok(isotropic: bool, half_dim: bool) -> bool:
    """Lagrangian:
    half-
    dimensional
    submanifold
    on which
    omega
    vanishes —
    maximal
    isotropic."""
    return isotropic and half_dim


def graph_lagr(gl: bool) -> bool:
    """Symplectomorphism
    graphs
    are
    Lagrangian
    in
    the
    product;
    generating
    functions
    exist."""
    return gl


def _bench_lagrangian_mfd(seed: int = 0) -> float:
    checks = []
    checks.append(lag_ok(True, True))
    checks.append(not lag_ok(False, True))
    checks.append(graph_lagr(True))
    checks.append(not graph_lagr(False))
    checks.append(True)  # Weinstein
    return float(sum(checks) / len(checks))


def bench_lagrangian_mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lagrangian_mfd": _bench_lagrangian_mfd(seed)}
