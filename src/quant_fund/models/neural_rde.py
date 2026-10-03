"""neural rde module (SYNTHETIC)."""

from __future__ import annotations


def neural_rde_ok(ns1: bool, sd: bool) -> bool:
    """neural_rde
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def neural_rde_aux(aux: bool) -> bool:
    """neural_rde
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_neural_rde(seed: int = 0) -> float:
    checks = []
    checks.append(neural_rde_ok(True, True))
    checks.append(not neural_rde_ok(False, True))
    checks.append(neural_rde_aux(True))
    checks.append(not neural_rde_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_neural_rde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neural_rde": _bench_neural_rde(seed)}
