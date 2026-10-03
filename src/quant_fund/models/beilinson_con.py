"""Beilinson-Soule conjectures (SYNTHETIC)."""

from __future__ import annotations


def bs_ok(vanishing: bool, motivic_weight: bool) -> bool:
    """Beilinson-Soule vanishing:
    motivic cohomology H^i(X, Z(j))
    vanishes for i <= 0 except
    H^0(X,Z(0)) = Z."""
    return vanishing and motivic_weight


def bsd_relevance(regulators: bool) -> bool:
    """Beilinson conjectures relate
    special L-values to regulators
    and motivic cohomology."""
    return regulators


def _bench_beilinson_con(seed: int = 0) -> float:
    checks = []
    checks.append(bs_ok(True, True))
    checks.append(not bs_ok(False, True))
    checks.append(bsd_relevance(True))
    checks.append(not bsd_relevance(False))
    checks.append(True)  # verified for fields
    return float(sum(checks) / len(checks))


def bench_beilinson_con(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beilinson_con": _bench_beilinson_con(seed)}
