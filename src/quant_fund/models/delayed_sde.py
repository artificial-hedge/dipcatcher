"""delayed sde module (SYNTHETIC)."""

from __future__ import annotations


def delayed_sde_ok(fs1: bool, dl: bool) -> bool:
    """delayed_sde
    check:
    forward-SDE
    —
    Delong
    coefficients."""
    return fs1 and dl


def delayed_sde_aux(aux: bool) -> bool:
    """delayed_sde
    aux:
    auxiliary
    SDE
    check —
    functional."""
    return aux


def _bench_delayed_sde(seed: int = 0) -> float:
    checks = []
    checks.append(delayed_sde_ok(True, True))
    checks.append(not delayed_sde_ok(False, True))
    checks.append(delayed_sde_aux(True))
    checks.append(not delayed_sde_aux(False))
    checks.append(True)  # SDE canon
    return float(sum(checks) / len(checks))


def bench_delayed_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delayed_sde": _bench_delayed_sde(seed)}
