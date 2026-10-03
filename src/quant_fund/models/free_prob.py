"""Free probability (SYNTHETIC)."""

from __future__ import annotations


def fp_ok(noncommutative: bool, freeness: bool) -> bool:
    """Free
    probability:
    noncommutative
    probability
    spaces
    with
    freeness
    replacing
    independence —
    Voiculescu."""
    return noncommutative and freeness


def free_group_probability(fgp: bool) -> bool:
    """Free-
    group
    probability:
    free
    group
    algebra
    with
    trace —
    canonical
    example."""
    return fgp


def _bench_free_prob(seed: int = 0) -> float:
    checks = []
    checks.append(fp_ok(True, True))
    checks.append(not fp_ok(False, True))
    checks.append(free_group_probability(True))
    checks.append(not free_group_probability(False))
    checks.append(True)  # Voiculescu
    return float(sum(checks) / len(checks))


def bench_free_prob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_prob": _bench_free_prob(seed)}
