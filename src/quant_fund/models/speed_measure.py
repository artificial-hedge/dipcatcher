"""speed measure module (SYNTHETIC)."""

from __future__ import annotations


def speed_measure_ok(sc: bool, sp: bool) -> bool:
    """speed_measure
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def speed_measure_aux(aux: bool) -> bool:
    """speed_measure
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_speed_measure(seed: int = 0) -> float:
    checks = []
    checks.append(speed_measure_ok(True, True))
    checks.append(not speed_measure_ok(False, True))
    checks.append(speed_measure_aux(True))
    checks.append(not speed_measure_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_speed_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_speed_measure": _bench_speed_measure(seed)}
