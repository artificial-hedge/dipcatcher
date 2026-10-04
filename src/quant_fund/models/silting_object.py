"""silting object module (SYNTHETIC)."""

from __future__ import annotations


def silting_object_ok(cluster: bool, quiver: bool) -> bool:
    """silting_object
    check:
    cluster
    structure —
    mutation."""
    return cluster and quiver


def silting_object_aux(aux: bool) -> bool:
    """silting_object
    aux:
    auxiliary
    cluster
    check —
    tilting."""
    return aux


def _bench_silting_object(seed: int = 0) -> float:
    checks = []
    checks.append(silting_object_ok(True, True))
    checks.append(not silting_object_ok(False, True))
    checks.append(silting_object_aux(True))
    checks.append(not silting_object_aux(False))
    checks.append(True)  # cluster canon
    return float(sum(checks) / len(checks))


def bench_silting_object(seed: int = 0) -> dict[str, float]:
    return {"synthetic_silting_object": _bench_silting_object(seed)}
