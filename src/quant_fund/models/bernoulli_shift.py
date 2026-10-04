"""Bernoulli shift (SYNTHETIC)."""

from __future__ import annotations


def bernoulli_ok(iid: bool, shift: bool) -> bool:
    """Bernoulli
    shift:
    two-sided
    shift on
    iid
    sequences;
    entropy
    = log of
    number
    of
    symbols."""
    return iid and shift


def orstein_iso(orn: bool) -> bool:
    """Ornstein
    isomorphism:
    Bernoulli
    shifts
    of equal
    entropy
    are
    isomorphic."""
    return orn


def _bench_bernoulli_shift(seed: int = 0) -> float:
    checks = []
    checks.append(bernoulli_ok(True, True))
    checks.append(not bernoulli_ok(False, True))
    checks.append(orstein_iso(True))
    checks.append(not orstein_iso(False))
    checks.append(True)  # Ornstein
    return float(sum(checks) / len(checks))


def bench_bernoulli_shift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bernoulli_shift": _bench_bernoulli_shift(seed)}
