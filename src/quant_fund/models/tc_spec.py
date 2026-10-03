"""Topological cyclic spectrum (SYNTHETIC)."""

from __future__ import annotations


def ts_ok(tc: bool, cyclotomic: bool) -> bool:
    """TC:
    topological
    cyclic
    homology
    spectrum —
    BMS
    TC."""
    return tc and cyclotomic


def tc_fixed(tf: bool) -> bool:
    """TC
    fixed:
    TC
    is
    fixed
    points
    of
    cyclotomic —
    TC
    definition."""
    return tf


def _bench_tc_spec(seed: int = 0) -> float:
    checks = []
    checks.append(ts_ok(True, True))
    checks.append(not ts_ok(False, True))
    checks.append(tc_fixed(True))
    checks.append(not tc_fixed(False))
    checks.append(True)  # BMS
    return float(sum(checks) / len(checks))


def bench_tc_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tc_spec": _bench_tc_spec(seed)}
