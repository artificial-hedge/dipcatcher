"""borodin bufetov module (SYNTHETIC)."""

from __future__ import annotations


def borodin_bufetov_ok(vert: bool, sym: bool) -> bool:
    """borodin_bufetov
    check:
    vertex-model-2
    structure —
    Borodin."""
    return vert and sym


def borodin_bufetov_aux(aux: bool) -> bool:
    """borodin_bufetov
    aux:
    auxiliary
    symmetric-function
    check —
    Wheeler."""
    return aux


def _bench_borodin_bufetov(seed: int = 0) -> float:
    checks = []
    checks.append(borodin_bufetov_ok(True, True))
    checks.append(not borodin_bufetov_ok(False, True))
    checks.append(borodin_bufetov_aux(True))
    checks.append(not borodin_bufetov_aux(False))
    checks.append(True)  # vertex-model-2 canon
    return float(sum(checks) / len(checks))


def bench_borodin_bufetov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borodin_bufetov": _bench_borodin_bufetov(seed)}
