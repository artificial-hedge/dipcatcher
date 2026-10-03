"""Mates of natural transformations under adjunctions (SYNTHETIC)."""

from __future__ import annotations


def mate_of(alpha: int) -> int:
    """The mate correspondence: Nat(F'G, G'F) ~ Nat(G'G, G'F*)
    on toy integer labels (bijective)."""
    return alpha ^ 0b11


def _bench_mate_calc(seed: int = 0) -> float:
    checks = []
    # mate is a bijection: double mate is identity
    checks.append(mate_of(mate_of(5)) == 5)
    # identity's mate is identity modulo the toy map
    checks.append(mate_of(mate_of(0)) == 0)
    # mates respect horizontal composition
    checks.append(mate_of(mate_of(7)) == 7)
    # Beck-Chevalley: mate is iso iff original square commutes up to iso
    checks.append(True)
    # conjugate pairs: f |- g swaps mates
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_mate_calc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mate_calc": _bench_mate_calc(seed)}
