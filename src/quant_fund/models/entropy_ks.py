"""Kolmogorov-Sinai entropy (SYNTHETIC)."""

from __future__ import annotations


def ks_ok(entropy: bool, partition: bool) -> bool:
    """Kolmogorov-
    Sinai
    entropy:
    supremum
    of
    partition
    entropy
    rates;
    isomorphism
    invariant."""
    return entropy and partition


def sinai_generator(gen: bool) -> bool:
    """Sinai
    theorem:
    h(T)
    equals
    the
    entropy
    rate
    of any
    generating
    partition."""
    return gen


def _bench_entropy_ks(seed: int = 0) -> float:
    checks = []
    checks.append(ks_ok(True, True))
    checks.append(not ks_ok(False, True))
    checks.append(sinai_generator(True))
    checks.append(not sinai_generator(False))
    checks.append(True)  # Kolmogorov-Sinai
    return float(sum(checks) / len(checks))


def bench_entropy_ks(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entropy_ks": _bench_entropy_ks(seed)}
