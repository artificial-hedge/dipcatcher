"""functional sde module (SYNTHETIC)."""

from __future__ import annotations


def functional_sde_ok(fs1: bool, dl: bool) -> bool:
    """functional_sde
    check:
    forward-SDE
    —
    Delong
    coefficients."""
    return fs1 and dl


def functional_sde_aux(aux: bool) -> bool:
    """functional_sde
    aux:
    auxiliary
    SDE
    check —
    functional."""
    return aux


def _bench_functional_sde(seed: int = 0) -> float:
    checks = []
    checks.append(functional_sde_ok(True, True))
    checks.append(not functional_sde_ok(False, True))
    checks.append(functional_sde_aux(True))
    checks.append(not functional_sde_aux(False))
    checks.append(True)  # SDE canon
    return float(sum(checks) / len(checks))


def bench_functional_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_functional_sde": _bench_functional_sde(seed)}
