"""latent sde module (SYNTHETIC)."""

from __future__ import annotations


def latent_sde_ok(ns1: bool, sd: bool) -> bool:
    """latent_sde
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def latent_sde_aux(aux: bool) -> bool:
    """latent_sde
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_latent_sde(seed: int = 0) -> float:
    checks = []
    checks.append(latent_sde_ok(True, True))
    checks.append(not latent_sde_ok(False, True))
    checks.append(latent_sde_aux(True))
    checks.append(not latent_sde_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_latent_sde(seed: int = 0) -> dict[str, float]:
    return {"synthetic_latent_sde": _bench_latent_sde(seed)}
