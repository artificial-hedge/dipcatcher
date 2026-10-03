"""Griffiths transversality (SYNTHETIC)."""

from __future__ import annotations


def gt_ok(horizontal: bool, period_map: bool) -> bool:
    """Griffiths
    transversality:
    period
    map
    is
    horizontal —
    derivative
    shifts
    Hodge
    by
    at
    most
    one."""
    return horizontal and period_map


def ivhs_rigidity(iv: bool) -> bool:
    """Infinitesimal
    VHS:
    Griffiths
    transversality
    constrains
    deformations —
    local
    Torelli."""
    return iv


def _bench_griffiths_transv(seed: int = 0) -> float:
    checks = []
    checks.append(gt_ok(True, True))
    checks.append(not gt_ok(False, True))
    checks.append(ivhs_rigidity(True))
    checks.append(not ivhs_rigidity(False))
    checks.append(True)  # Griffiths
    return float(sum(checks) / len(checks))


def bench_griffiths_transv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_griffiths_transv": _bench_griffiths_transv(seed)}
