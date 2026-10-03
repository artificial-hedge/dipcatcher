"""Rankin-Selberg L-functions (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(convolution: bool, integral: bool) -> bool:
    """Rankin-
    Selberg
    convolution:
    L(s, pi x
    pi')
    via the
    unfolding
    integral;
    functional
    equation."""
    return convolution and integral


def functorial_prod(prod: bool) -> bool:
    """Rankin
    product
    detects
    functorial
    transfer
    GL_m x
    GL_n ->
    GL_{mn}."""
    return prod


def _bench_rankin_selberg(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(functorial_prod(True))
    checks.append(not functorial_prod(False))
    checks.append(True)  # Rankin-Selberg
    return float(sum(checks) / len(checks))


def bench_rankin_selberg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rankin_selberg": _bench_rankin_selberg(seed)}
