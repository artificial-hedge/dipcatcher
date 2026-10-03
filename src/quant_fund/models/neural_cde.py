"""neural cde module (SYNTHETIC)."""

from __future__ import annotations


def neural_cde_ok(ns1: bool, sd: bool) -> bool:
    """neural_cde
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def neural_cde_aux(aux: bool) -> bool:
    """neural_cde
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_neural_cde(seed: int = 0) -> float:
    checks = []
    checks.append(neural_cde_ok(True, True))
    checks.append(not neural_cde_ok(False, True))
    checks.append(neural_cde_aux(True))
    checks.append(not neural_cde_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_neural_cde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neural_cde": _bench_neural_cde(seed)}
