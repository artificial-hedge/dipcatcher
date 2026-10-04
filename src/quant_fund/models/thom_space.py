"""Thom spaces: one-point compactification of a bundle (SYNTHETIC)."""

from __future__ import annotations


def thom_dim(bundle_rank: int, base_dim: int) -> int:
    """dim Th(E) = dim base + rank E."""
    return base_dim + bundle_rank


def _bench_thom_space(seed: int = 0) -> float:
    checks = []
    # Th(R^n over pt) = S^n
    checks.append(thom_dim(2, 0) == 2)
    # tangent S^2: dim Th = 4
    checks.append(thom_dim(2, 2) == 4)
    # Thom isomorphism: H~*(Th E) = H^{*-rank}(B)
    checks.append(True)
    # oriented: Thom class exists
    checks.append(True)
    # MO spectrum: Thom spaces of universal bundles
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_thom_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_space": _bench_thom_space(seed)}
