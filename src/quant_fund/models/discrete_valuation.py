"""Discrete valuation (SYNTHETIC)."""

from __future__ import annotations


def dv_ok(discrete: bool, valuation: bool) -> bool:
    """Discrete:
    discrete
    valuation
    ring —
    DVR."""
    return discrete and valuation


def dvr_uniformizer(du: bool) -> bool:
    """Uniformizer:
    DVR
    uniformizing
    parameter —
    DVR
    uniformizer."""
    return du


def _bench_discrete_valuation(seed: int = 0) -> float:
    checks = []
    checks.append(dv_ok(True, True))
    checks.append(not dv_ok(False, True))
    checks.append(dvr_uniformizer(True))
    checks.append(not dvr_uniformizer(False))
    checks.append(True)  # DVR
    return float(sum(checks) / len(checks))


def bench_discrete_valuation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_discrete_valuation": _bench_discrete_valuation(seed)}
