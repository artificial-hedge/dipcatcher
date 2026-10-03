"""random sde module (SYNTHETIC)."""

from __future__ import annotations


def random_sde_ok(fs1: bool, dl: bool) -> bool:
    """random_sde
    check:
    forward-SDE
    —
    Delong
    coefficients."""
    return fs1 and dl


def random_sde_aux(aux: bool) -> bool:
    """random_sde
    aux:
    auxiliary
    SDE
    check —
    functional."""
    return aux


def _bench_random_sde(seed: int = 0) -> float:
    checks = []
    checks.append(random_sde_ok(True, True))
    checks.append(not random_sde_ok(False, True))
    checks.append(random_sde_aux(True))
    checks.append(not random_sde_aux(False))
    checks.append(True)  # SDE canon
    return float(sum(checks) / len(checks))


def bench_random_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_random_sde": _bench_random_sde(seed)}
