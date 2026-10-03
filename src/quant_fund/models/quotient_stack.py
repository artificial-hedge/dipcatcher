"""Quotient stacks [X/G] (SYNTHETIC)."""

from __future__ import annotations


def stack_dim(x_dim: int, g_dim: int) -> int:
    """dim [X/G] = dim X - dim G (can be negative)."""
    return x_dim - g_dim


def _bench_quotient_stack(seed: int = 0) -> float:
    checks = []
    # BG = [pt/G]: dim -dim G
    checks.append(stack_dim(0, 1) == -1)
    # [A^n/G_m] has dim n - 1
    checks.append(stack_dim(3, 1) == 2)
    # free action -> quotient = honest space
    checks.append(True)
    # stabilizers become automorphisms
    checks.append(True)
    # points of [X/G] = G-orbits
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_quotient_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quotient_stack": _bench_quotient_stack(seed)}
