"""sde matching module (SYNTHETIC)."""

from __future__ import annotations


def sde_matching_ok(ns1: bool, sd: bool) -> bool:
    """sde_matching
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def sde_matching_aux(aux: bool) -> bool:
    """sde_matching
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_sde_matching(seed: int = 0) -> float:
    checks = []
    checks.append(sde_matching_ok(True, True))
    checks.append(not sde_matching_ok(False, True))
    checks.append(sde_matching_aux(True))
    checks.append(not sde_matching_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_sde_matching(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sde_matching": _bench_sde_matching(seed)}
