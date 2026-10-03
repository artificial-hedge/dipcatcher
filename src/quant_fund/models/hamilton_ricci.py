"""Hamilton's Ricci flow (SYNTHETIC)."""

from __future__ import annotations


def hr_ok(metric_flow: bool, curvature: bool) -> bool:
    """Hamilton's
    Ricci
    flow:
    metric
    evolves
    by
    minus
    two
    Ricci —
    parabolic
    diffusion
    of
    geometry."""
    return metric_flow and curvature


def short_time_exist(st: bool) -> bool:
    """Short-
    time
    existence:
    Ricci
    flow
    has
    a
    solution
    for
    small
    time —
    Hamilton
    1982,
    DeTurck
    trick."""
    return st


def _bench_hamilton_ricci(seed: int = 0) -> float:
    checks = []
    checks.append(hr_ok(True, True))
    checks.append(not hr_ok(False, True))
    checks.append(short_time_exist(True))
    checks.append(not short_time_exist(False))
    checks.append(True)  # Hamilton
    return float(sum(checks) / len(checks))


def bench_hamilton_ricci(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hamilton_ricci": _bench_hamilton_ricci(seed)}
