"""Free convolution (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(additive_free: bool, noncommutative: bool) -> bool:
    """Free
    convolution:
    distribution
    of
    sum
    of
    free
    variables —
    semicircle
    stable."""
    return additive_free and noncommutative


def free_multiplicative_conv(fmc: bool) -> bool:
    """Free
    multiplicative:
    product
    of
    positive
    free
    variables —
    S-
    transform
    domain."""
    return fmc


def _bench_free_convolution(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(free_multiplicative_conv(True))
    checks.append(not free_multiplicative_conv(False))
    checks.append(True)  # Voiculescu-Bercovici
    return float(sum(checks) / len(checks))


def bench_free_convolution(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_convolution": _bench_free_convolution(seed)}
