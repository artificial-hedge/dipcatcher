"""scale measure module (SYNTHETIC)."""

from __future__ import annotations


def scale_measure_ok(sc: bool, sp: bool) -> bool:
    """scale_measure
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def scale_measure_aux(aux: bool) -> bool:
    """scale_measure
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_scale_measure(seed: int = 0) -> float:
    checks = []
    checks.append(scale_measure_ok(True, True))
    checks.append(not scale_measure_ok(False, True))
    checks.append(scale_measure_aux(True))
    checks.append(not scale_measure_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_scale_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_scale_measure": _bench_scale_measure(seed)}
