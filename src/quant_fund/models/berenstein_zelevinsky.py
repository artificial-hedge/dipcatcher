"""Berenstein-Zelevinsky triangles (SYNTHETIC)."""

from __future__ import annotations


def bz_ok(triangles: bool, tensor_products: bool) -> bool:
    """Berenstein-
    Zelevinsky:
    triangular
    arrays
    parametrizing
    tensor
    product
    multiplicities —
    polyhedral
    model."""
    return triangles and tensor_products


def hive_condition(hc: bool) -> bool:
    """Hive
    condition:
    linear
    inequalities
    defining
    LR
    cones —
    BZ
    triangles
    realize
    it."""
    return hc


def _bench_berenstein_zelevinsky(seed: int = 0) -> float:
    checks = []
    checks.append(bz_ok(True, True))
    checks.append(not bz_ok(False, True))
    checks.append(hive_condition(True))
    checks.append(not hive_condition(False))
    checks.append(True)  # Berenstein-Zelevinsky
    return float(sum(checks) / len(checks))


def bench_berenstein_zelevinsky(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berenstein_zelevinsky": _bench_berenstein_zelevinsky(seed)}
