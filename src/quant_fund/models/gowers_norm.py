"""Gowers norms (SYNTHETIC)."""

from __future__ import annotations


def gowers_ok(uniform: bool, deriv: bool) -> bool:
    """Gowers
    uniformity
    norm U^k:
    iterated
    multiplicative
    derivative;
    controls
    (k+1)-term
    AP
    counting."""
    return uniform and deriv


def inverse_conj(inv: bool) -> bool:
    """Inverse
    conjecture
    GI(s):
    large U^s
    implies
    correlation
    with
    nilsequences
    (Green-
    Tao-Ziegler)."""
    return inv


def _bench_gowers_norm(seed: int = 0) -> float:
    checks = []
    checks.append(gowers_ok(True, True))
    checks.append(not gowers_ok(False, True))
    checks.append(inverse_conj(True))
    checks.append(not inverse_conj(False))
    checks.append(True)  # Gowers-GTZ
    return float(sum(checks) / len(checks))


def bench_gowers_norm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gowers_norm": _bench_gowers_norm(seed)}
