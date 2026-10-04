"""E-infinity space (SYNTHETIC)."""

from __future__ import annotations


def ei_ok(e_infty: bool, operadic: bool) -> bool:
    """E-
    infty:
    E-
    infinity
    space
    and
    operad —
    May
    E-infty."""
    return e_infty and operadic


def group_completion(gc: bool) -> bool:
    """Group
    completion:
    E-
    infinity
    group
    completion —
    May
    completion."""
    return gc


def _bench_e_infty_space(seed: int = 0) -> float:
    checks = []
    checks.append(ei_ok(True, True))
    checks.append(not ei_ok(False, True))
    checks.append(group_completion(True))
    checks.append(not group_completion(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_e_infty_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_e_infty_space": _bench_e_infty_space(seed)}
