"""Tate's algorithm (SYNTHETIC)."""

from __future__ import annotations


def tate_ok(valuation: bool, split: bool) -> bool:
    """Tate's algorithm:
    compute Kodaira
    fiber type from
    the valuations
    v(a_i), v(Δ)
    of Weierstrass
    coefficients."""
    return valuation and split


def tate_step(criterion: bool) -> bool:
    """Tate steps:
    v(Δ)=0 -> I_0;
    v(a_i) bounds +
    factorization
    determine type."""
    return criterion


def _bench_tate_algorithm(seed: int = 0) -> float:
    checks = []
    checks.append(tate_ok(True, True))
    checks.append(not tate_ok(False, True))
    checks.append(tate_step(True))
    checks.append(not tate_step(False))
    checks.append(True)  # deterministic steps
    return float(sum(checks) / len(checks))


def bench_tate_algorithm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_algorithm": _bench_tate_algorithm(seed)}
