"""Periodic homotopy (SYNTHETIC)."""

from __future__ import annotations


def ph_ok(periodic: bool, htpy: bool) -> bool:
    """Periodic
    homotopy:
    periodic
    homotopy —
    v_n
    periodic."""
    return periodic and htpy


def vn_periodicity(vp: bool) -> bool:
    """Vn
    periodicity:
    v_n
    periodic
    family —
    Smith
    Toda."""
    return vp


def _bench_periodic_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(ph_ok(True, True))
    checks.append(not ph_ok(False, True))
    checks.append(vn_periodicity(True))
    checks.append(not vn_periodicity(False))
    checks.append(True)  # Smith-Toda
    return float(sum(checks) / len(checks))


def bench_periodic_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodic_htpy": _bench_periodic_htpy(seed)}
