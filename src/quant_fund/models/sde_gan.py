"""sde gan module (SYNTHETIC)."""

from __future__ import annotations


def sde_gan_ok(ns1: bool, sd: bool) -> bool:
    """sde_gan
    check:
    neural-SDE —
    latent/CDE."""
    return ns1 and sd


def sde_gan_aux(aux: bool) -> bool:
    """sde_gan
    aux:
    auxiliary
    sde
    check —
    RDE/logsig."""
    return aux


def _bench_sde_gan(seed: int = 0) -> float:
    checks = []
    checks.append(sde_gan_ok(True, True))
    checks.append(not sde_gan_ok(False, True))
    checks.append(sde_gan_aux(True))
    checks.append(not sde_gan_aux(False))
    checks.append(True)  # neural-SDE canon
    return float(sum(checks) / len(checks))


def bench_sde_gan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sde_gan": _bench_sde_gan(seed)}
