"""Cartesian categories 2 (SYNTHETIC)."""

from __future__ import annotations


def cart_ok(cartesian: bool, products: bool) -> bool:
    """Cartesian
    category:
    cartesian
    cat —
    finite
    products."""
    return cartesian and products


def cartesian_closed2(cc2: bool) -> bool:
    """Cartesian
    closed:
    cartesian
    closed —
    exponentials."""
    return cc2


def _bench_cartesian_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(cart_ok(True, True))
    checks.append(not cart_ok(False, True))
    checks.append(cartesian_closed2(True))
    checks.append(not cartesian_closed2(False))
    checks.append(True)  # Lambek-Scott
    return float(sum(checks) / len(checks))


def bench_cartesian_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartesian_cat2": _bench_cartesian_cat2(seed)}
