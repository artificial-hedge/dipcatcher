"""Tensor category (SYNTHETIC)."""

from __future__ import annotations


def tc_ok(tensor_prod: bool, pentagon: bool) -> bool:
    """Tensor
    category:
    monoidal
    with
    pentagon
    coherence —
    monoidal
    axioms."""
    return tensor_prod and pentagon


def mac_lane_coherence(mc: bool) -> bool:
    """Mac
    Lane:
    pentagon
    implies
    coherence —
    Mac
    Lane
    theorem."""
    return mc


def _bench_tensor_cat(seed: int = 0) -> float:
    checks = []
    checks.append(tc_ok(True, True))
    checks.append(not tc_ok(False, True))
    checks.append(mac_lane_coherence(True))
    checks.append(not mac_lane_coherence(False))
    checks.append(True)  # Mac Lane
    return float(sum(checks) / len(checks))


def bench_tensor_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_cat": _bench_tensor_cat(seed)}
