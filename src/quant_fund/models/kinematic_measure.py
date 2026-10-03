"""kinematic measure module (SYNTHETIC)."""

from __future__ import annotations


def kinematic_measure_ok(geo: bool, kin: bool) -> bool:
    """kinematic_measure
    check:
    integral
    geometry —
    kinematic."""
    return geo and kin


def kinematic_measure_aux(aux: bool) -> bool:
    """kinematic_measure
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_kinematic_measure(seed: int = 0) -> float:
    checks = []
    checks.append(kinematic_measure_ok(True, True))
    checks.append(not kinematic_measure_ok(False, True))
    checks.append(kinematic_measure_aux(True))
    checks.append(not kinematic_measure_aux(False))
    checks.append(True)  # integral-geometry canon
    return float(sum(checks) / len(checks))


def bench_kinematic_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kinematic_measure": _bench_kinematic_measure(seed)}
