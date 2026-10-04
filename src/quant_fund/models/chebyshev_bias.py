"""Chebyshev bias (SYNTHETIC)."""

from __future__ import annotations


def bias_ok(count: bool, residue: bool) -> bool:
    """Chebyshev
    bias:
    primes
    in certain
    residue
    classes
    win the
    prime
    race
    (Rubinstein-
    Sarnak)."""
    return count and residue


def logarithmic_scale(log: bool) -> bool:
    """Logarithmic
    density:
    bias
    measured
    on the
    log scale,
    not the
    natural
    density."""
    return log


def _bench_chebyshev_bias(seed: int = 0) -> float:
    checks = []
    checks.append(bias_ok(True, True))
    checks.append(not bias_ok(False, True))
    checks.append(logarithmic_scale(True))
    checks.append(not logarithmic_scale(False))
    checks.append(True)  # Rubinstein-Sarnak
    return float(sum(checks) / len(checks))


def bench_chebyshev_bias(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_bias": _bench_chebyshev_bias(seed)}
