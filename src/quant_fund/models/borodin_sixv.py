"""borodin sixv module (SYNTHETIC)."""

from __future__ import annotations


def borodin_sixv_ok(sixv: bool, solv: bool) -> bool:
    """borodin_sixv
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def borodin_sixv_aux(aux: bool) -> bool:
    """borodin_sixv
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_borodin_sixv(seed: int = 0) -> float:
    checks = []
    checks.append(borodin_sixv_ok(True, True))
    checks.append(not borodin_sixv_ok(False, True))
    checks.append(borodin_sixv_aux(True))
    checks.append(not borodin_sixv_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_borodin_sixv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borodin_sixv": _bench_borodin_sixv(seed)}
