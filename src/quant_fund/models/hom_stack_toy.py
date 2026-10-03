"""Hom schemes: morphisms P^1 -> P^1 of degree d (SYNTHETIC)."""

from __future__ import annotations


def hom_dim(deg: int) -> int:
    """Hom_d(P^1, P^1) = P^{2d+1}: pairs of degree-d binary forms up to
    scale -> dimension 2d+1."""
    return 2 * deg + 1


def _bench_hom_stack_toy(seed: int = 0) -> float:
    checks = []
    # degree-1 maps = PGL_2, dim 3
    checks.append(hom_dim(1) == 3)
    # degree-2 rational maps, dim 5
    checks.append(hom_dim(2) == 5)
    # degree-0 = constant maps = target, dim 1
    checks.append(hom_dim(0) == 1)
    # dim grows linearly
    checks.append(hom_dim(3) - hom_dim(2) == 2)
    return float(sum(checks) / len(checks))


def bench_hom_stack_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hom_stack_toy": _bench_hom_stack_toy(seed)}
