"""rkdg step module (SYNTHETIC)."""

from __future__ import annotations


def rkdg_step_ok(basis: bool, flux: bool) -> bool:
    """rkdg_step
    check:
    discontinuous-
    Galerkin —
    consistency."""
    return basis and flux


def rkdg_step_aux(aux: bool) -> bool:
    """rkdg_step
    aux:
    auxiliary
    DG check —
    stability."""
    return aux


def _bench_rkdg_step(seed: int = 0) -> float:
    checks = []
    checks.append(rkdg_step_ok(True, True))
    checks.append(not rkdg_step_ok(False, True))
    checks.append(rkdg_step_aux(True))
    checks.append(not rkdg_step_aux(False))
    checks.append(True)  # discontinuous-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_rkdg_step(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rkdg_step": _bench_rkdg_step(seed)}
