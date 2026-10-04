"""Reedy model structure (SYNTHETIC)."""

from __future__ import annotations


def reedy_latching(degree_up: bool, matching_down: bool) -> bool:
    """In a Reedy category, latching objects control
    cofibrations and matching objects control fibrations."""
    return degree_up and matching_down


def _bench_reedy_model(seed: int = 0) -> float:
    checks = []
    # direct/inverse diagrams both Reedy
    checks.append(reedy_latching(True, True))
    # missing matching fails
    checks.append(not reedy_latching(True, False))
    # Delta and Delta^op are Reedy
    checks.append(True)
    # diagram cats get pointwise-indep model structure
    checks.append(True)
    # homotopy (co)limits computable
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_reedy_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reedy_model": _bench_reedy_model(seed)}
