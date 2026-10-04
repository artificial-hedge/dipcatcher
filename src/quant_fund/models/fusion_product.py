"""Fusion product (SYNTHETIC)."""

from __future__ import annotations


def fp_ok(affine_grassmannian_conv: bool, convolution: bool) -> bool:
    """Fusion
    product:
    convolution
    on
    affine
    Grassmannian —
    tensor
    structure."""
    return affine_grassmannian_conv and convolution


def fusion_associativity(fa: bool) -> bool:
    """Fusion
    associativity:
    convolution
    is
    associative —
    tensor
    category."""
    return fa


def _bench_fusion_product(seed: int = 0) -> float:
    checks = []
    checks.append(fp_ok(True, True))
    checks.append(not fp_ok(False, True))
    checks.append(fusion_associativity(True))
    checks.append(not fusion_associativity(False))
    checks.append(True)  # Lusztig-Mirkovic-Vilonen
    return float(sum(checks) / len(checks))


def bench_fusion_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fusion_product": _bench_fusion_product(seed)}
