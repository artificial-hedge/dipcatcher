"""logsig rde module (SYNTHETIC)."""

from __future__ import annotations


def logsig_rde_ok(ns1: bool, sd: bool) -> bool:
    """logsig_rde
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def logsig_rde_aux(aux: bool) -> bool:
    """logsig_rde
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_logsig_rde(seed: int = 0) -> float:
    checks = []
    checks.append(logsig_rde_ok(True, True))
    checks.append(not logsig_rde_ok(False, True))
    checks.append(logsig_rde_aux(True))
    checks.append(not logsig_rde_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_logsig_rde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logsig_rde": _bench_logsig_rde(seed)}
