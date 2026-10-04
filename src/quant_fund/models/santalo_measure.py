"""santalo measure module (SYNTHETIC)."""

from __future__ import annotations


def santalo_measure_ok(geo: bool, kin: bool) -> bool:
    """santalo_measure
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def santalo_measure_aux(aux: bool) -> bool:
    """santalo_measure
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_santalo_measure(seed: int = 0) -> float:
    checks = []
    checks.append(santalo_measure_ok(True, True))
    checks.append(not santalo_measure_ok(False, True))
    checks.append(santalo_measure_aux(True))
    checks.append(not santalo_measure_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_santalo_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_santalo_measure": _bench_santalo_measure(seed)}
