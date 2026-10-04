"""Periodic cyclic (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(periodic: bool, cyclic: bool) -> bool:
    """Periodic:
    periodic
    cyclic
    homology —
    TP
    homology."""
    return periodic and cyclic


def tp_bms(tpb: bool) -> bool:
    """TP
    BMS:
    periodic
    cyclic
    vs
    THH
    —
    BMS
    TP."""
    return tpb


def _bench_periodic_cyclic(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(tp_bms(True))
    checks.append(not tp_bms(False))
    checks.append(True)  # BMS
    return float(sum(checks) / len(checks))


def bench_periodic_cyclic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodic_cyclic": _bench_periodic_cyclic(seed)}
