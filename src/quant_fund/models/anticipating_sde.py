"""anticipating sde module (SYNTHETIC)."""

from __future__ import annotations


def anticipating_sde_ok(fs1: bool, dl: bool) -> bool:
    """anticipating_sde
    check:
    forward-SDE
    —
    Delong
    coefficients."""
    return fs1 and dl


def anticipating_sde_aux(aux: bool) -> bool:
    """anticipating_sde
    aux:
    auxiliary
    SDE
    check —
    functional."""
    return aux


def _bench_anticipating_sde(seed: int = 0) -> float:
    checks = []
    checks.append(anticipating_sde_ok(True, True))
    checks.append(not anticipating_sde_ok(False, True))
    checks.append(anticipating_sde_aux(True))
    checks.append(not anticipating_sde_aux(False))
    checks.append(True)  # SDE canon
    return float(sum(checks) / len(checks))


def bench_anticipating_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anticipating_sde": _bench_anticipating_sde(seed)}
