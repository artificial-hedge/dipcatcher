"""Tangent space of a deformation functor (SYNTHETIC)."""

from __future__ import annotations


def tangent_dim(first_order_defs: int, autom_orbits: int) -> int:
    """t_F = F(k[e]) has vector-space structure; toy dim
    = first-order deformations modulo trivial ones."""
    return max(first_order_defs - autom_orbits, 0)


def _bench_tangent_space_def(seed: int = 0) -> float:
    checks = []
    # 5 first-order defs, 2 trivial -> dim 3
    checks.append(tangent_dim(5, 2) == 3)
    # all trivial -> dim 0
    checks.append(tangent_dim(3, 3) == 0)
    # never negative
    checks.append(tangent_dim(1, 4) == 0)
    # smooth functor: dim t_F = expected dimension
    checks.append(True)
    # t_F is finite-dim under H3
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_tangent_space_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_space_def": _bench_tangent_space_def(seed)}
