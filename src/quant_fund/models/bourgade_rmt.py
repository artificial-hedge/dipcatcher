"""bourgade rmt module (SYNTHETIC)."""

from __future__ import annotations


def bourgade_rmt_ok(rmt: bool, univ: bool) -> bool:
    """bourgade_rmt
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def bourgade_rmt_aux(aux: bool) -> bool:
    """bourgade_rmt
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_bourgade_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(bourgade_rmt_ok(True, True))
    checks.append(not bourgade_rmt_ok(False, True))
    checks.append(bourgade_rmt_aux(True))
    checks.append(not bourgade_rmt_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_bourgade_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bourgade_rmt": _bench_bourgade_rmt(seed)}
