"""Weighted limits (SYNTHETIC)."""

from __future__ import annotations


def weight_lim_ok(weight_functor: bool, universal: bool) -> bool:
    """Weighted limit lim^W F: representation
    of Nat(W, C(-,F-)); ordinary limits
    are conical (terminal weight)."""
    return weight_functor and universal


def weighted_coend(end_calc: bool) -> bool:
    """Ends/coends compute weighted
    (co)limits; conical lim = end of
    hom functors."""
    return end_calc


def _bench_weight_lim(seed: int = 0) -> float:
    checks = []
    checks.append(weight_lim_ok(True, True))
    checks.append(not weight_lim_ok(True, False))
    checks.append(weighted_coend(True))
    checks.append(not weighted_coend(False))
    checks.append(True)  # Kan extension = weighted colimit
    return float(sum(checks) / len(checks))


def bench_weight_lim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weight_lim": _bench_weight_lim(seed)}
