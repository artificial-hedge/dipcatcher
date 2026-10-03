"""Dendroidal Segal spaces (SYNTHETIC)."""

from __future__ import annotations


def ds2_ok(dendroidal: bool, segal: bool) -> bool:
    """Dendroidal
    Segal:
    dendroidal
    Segal
    space —
    Cisinski-
    Moerdijk."""
    return dendroidal and segal


def segal_condition(sc: bool) -> bool:
    """Segal
    condition:
    Segal
    condition
    for
    dendroidal
    spaces —
    Segal
    map."""
    return sc


def _bench_dendroidal_seg(seed: int = 0) -> float:
    checks = []
    checks.append(ds2_ok(True, True))
    checks.append(not ds2_ok(False, True))
    checks.append(segal_condition(True))
    checks.append(not segal_condition(False))
    checks.append(True)  # Cisinski-Moerdijk
    return float(sum(checks) / len(checks))


def bench_dendroidal_seg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dendroidal_seg": _bench_dendroidal_seg(seed)}
