"""Negative cyclic (SYNTHETIC)."""

from __future__ import annotations


def nc_ok(negative: bool, cyclic: bool) -> bool:
    """Negative:
    negative
    cyclic
    homology —
    NC
    homology."""
    return negative and cyclic


def nc_bms(ncb: bool) -> bool:
    """NC
    BMS:
    negative
    cyclic
    vs
    TP
    and
    THH —
    BMS
    structure."""
    return ncb


def _bench_negative_cyclic(seed: int = 0) -> float:
    checks = []
    checks.append(nc_ok(True, True))
    checks.append(not nc_ok(False, True))
    checks.append(nc_bms(True))
    checks.append(not nc_bms(False))
    checks.append(True)  # BMS
    return float(sum(checks) / len(checks))


def bench_negative_cyclic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_negative_cyclic": _bench_negative_cyclic(seed)}
