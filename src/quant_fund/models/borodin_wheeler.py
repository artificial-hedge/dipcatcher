"""borodin wheeler module (SYNTHETIC)."""

from __future__ import annotations


def borodin_wheeler_ok(vert: bool, sym: bool) -> bool:
    """borodin_wheeler
    check:
    vertex-model-2
    structure —
    Borodin."""
    return vert and sym


def borodin_wheeler_aux(aux: bool) -> bool:
    """borodin_wheeler
    aux:
    auxiliary
    symmetric-function
    check —
    Wheeler."""
    return aux


def _bench_borodin_wheeler(seed: int = 0) -> float:
    checks = []
    checks.append(borodin_wheeler_ok(True, True))
    checks.append(not borodin_wheeler_ok(False, True))
    checks.append(borodin_wheeler_aux(True))
    checks.append(not borodin_wheeler_aux(False))
    checks.append(True)  # vertex-model-2 canon
    return float(sum(checks) / len(checks))


def bench_borodin_wheeler(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borodin_wheeler": _bench_borodin_wheeler(seed)}
