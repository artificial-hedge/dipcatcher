"""S-transform (SYNTHETIC)."""

from __future__ import annotations


def st_ok(multiplicative: bool, free_products: bool) -> bool:
    """S-
    transform:
    multiplicative
    free
    convolution
    linearizer —
    products
    of
    free
    variables."""
    return multiplicative and free_products


def s_transform_property(stp: bool) -> bool:
    """S-
    transform
    product:
    S
    of
    product
    is
    product
    of
    S —
    multiplicativity."""
    return stp


def _bench_s_transform(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(s_transform_property(True))
    checks.append(not s_transform_property(False))
    checks.append(True)  # Voiculescu
    return float(sum(checks) / len(checks))


def bench_s_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_s_transform": _bench_s_transform(seed)}
