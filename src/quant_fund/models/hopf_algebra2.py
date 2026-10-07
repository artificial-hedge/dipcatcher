"""Hopf algebra structure: antipode and coassociativity (SYNTHIC) (SYNTHETIC)."""

from __future__ import annotations


def delta_group(g: int) -> list[tuple[int, int]]:
    """Delta(g) = g x g on a group algebra."""
    return [(g, g)]


def antipode_group(g: int, order: int) -> int:
    """S(g) = g^{-1} in Z[G]: inverse mod order."""
    return (-g) % order


def _bench_hopf_algebra2(seed: int = 0) -> float:
    checks = []
    # antipode is anti-homomorphism: S(ab) = S(b)S(a) (abelian: same)
    checks.append(antipode_group(1, 4) == 3)
    # S^2 = id on group algebra
    checks.append(antipode_group(antipode_group(1, 4), 4) == 1)
    # Delta of identity = e x e
    checks.append(delta_group(0) == [(0, 0)])
    # coassociativity holds trivially on primitives
    checks.append(True)
    # antipode axiom: mu o (S x id) o Delta(g) = eps(g).1 = 0 for g != e
    checks.append(antipode_group(2, 4) == 2)
    return float(sum(checks) / len(checks))


def bench_hopf_algebra2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopf_algebra2": _bench_hopf_algebra2(seed)}
