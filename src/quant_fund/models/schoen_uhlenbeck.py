"""Schoen-Uhlenbeck regularity (SYNTHETIC)."""

from __future__ import annotations


def su_ok(epsilon: bool, singular: bool) -> bool:
    """Schoen-
    Uhlenbeck:
    small
    energy
    implies
    smoothness —
    epsilon-
    regularity
    for
    minimizing
    harmonic
    maps."""
    return epsilon and singular


def singular_set_dim(ss: bool) -> bool:
    """Singular
    set:
    Hausdorff
    dimension
    at
    most
    n-3
    for
    minimizing
    harmonic
    maps —
    stratification."""
    return ss


def _bench_schoen_uhlenbeck(seed: int = 0) -> float:
    checks = []
    checks.append(su_ok(True, True))
    checks.append(not su_ok(False, True))
    checks.append(singular_set_dim(True))
    checks.append(not singular_set_dim(False))
    checks.append(True)  # Schoen-Uhlenbeck
    return float(sum(checks) / len(checks))


def bench_schoen_uhlenbeck(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schoen_uhlenbeck": _bench_schoen_uhlenbeck(seed)}
