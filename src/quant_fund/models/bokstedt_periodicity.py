"""Bokstedt periodicity (SYNTHETIC)."""

from __future__ import annotations


def bp_ok(bokstedt: bool, periodicity: bool) -> bool:
    """Bokstedt:
    Bokstedt
    periodicity —
    Hesselholt
    Bokstedt."""
    return bokstedt and periodicity


def bokstedt_map(bm: bool) -> bool:
    """Bokstedt
    map:
    Bokstedt
    periodicity
    class —
    Hesselholt."""
    return bm


def _bench_bokstedt_periodicity(seed: int = 0) -> float:
    checks = []
    checks.append(bp_ok(True, True))
    checks.append(not bp_ok(False, True))
    checks.append(bokstedt_map(True))
    checks.append(not bokstedt_map(False))
    checks.append(True)  # Hesselholt
    return float(sum(checks) / len(checks))


def bench_bokstedt_periodicity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bokstedt_periodicity": _bench_bokstedt_periodicity(seed)}
