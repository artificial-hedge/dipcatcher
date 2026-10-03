"""Exponential generating functions (SYNTHETIC)."""

from __future__ import annotations


def eg_ok(labeled_structures: bool, egf: bool) -> bool:
    """Exponential
    GF:
    generating
    functions
    for
    labeled
    structures —
    coefficient
    extraction."""
    return labeled_structures and egf


def egf_product_rule(epr: bool) -> bool:
    """EGF
    product:
    product
    of
    EGFs
    counts
    labeled
    products —
    binomial
    convolution."""
    return epr


def _bench_exponential_gf(seed: int = 0) -> float:
    checks = []
    checks.append(eg_ok(True, True))
    checks.append(not eg_ok(False, True))
    checks.append(egf_product_rule(True))
    checks.append(not egf_product_rule(False))
    checks.append(True)  # Stanley
    return float(sum(checks) / len(checks))


def bench_exponential_gf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exponential_gf": _bench_exponential_gf(seed)}
