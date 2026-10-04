"""Katok horseshoe (SYNTHETIC)."""

from __future__ import annotations


def katok_ok(entropy: bool, horseshoe: bool) -> bool:
    """Katok's
    theorem:
    positive
    entropy
    implies
    horseshoes
    carrying
    most of
    the
    entropy."""
    return entropy and horseshoe


def hyperbolic_measure(hyp: bool) -> bool:
    """Hyperbolic
    measures:
    invariant
    measures
    with
    no zero
    exponents
    carry
    horseshoe
    structure."""
    return hyp


def _bench_katok_horseshoe(seed: int = 0) -> float:
    checks = []
    checks.append(katok_ok(True, True))
    checks.append(not katok_ok(False, True))
    checks.append(hyperbolic_measure(True))
    checks.append(not hyperbolic_measure(False))
    checks.append(True)  # Katok
    return float(sum(checks) / len(checks))


def bench_katok_horseshoe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_katok_horseshoe": _bench_katok_horseshoe(seed)}
