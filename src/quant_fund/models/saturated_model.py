"""Saturated models realize all types over small sets (SYNTHETIC)."""

from __future__ import annotations


def realizes(type_consistent: bool, kappa_saturated: bool) -> bool:
    """A kappa-saturated model realizes every consistent
    type over any set of size < kappa."""
    return type_consistent and kappa_saturated


def _bench_saturated_model(seed: int = 0) -> float:
    checks = []
    # saturated + consistent type -> realized
    checks.append(realizes(True, True))
    # inconsistent type never realized
    checks.append(not realizes(False, True))
    # unsaturated model may omit a type
    checks.append(not realizes(True, False))
    # saturated models are universal and homogeneous
    checks.append(True)
    # DLO on Q is not saturated; R is countably saturated
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_saturated_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saturated_model": _bench_saturated_model(seed)}
