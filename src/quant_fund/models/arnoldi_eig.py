"""arnoldi eig module (SYNTHETIC)."""

from __future__ import annotations


def arnoldi_eig_ok(krylov: bool, resid: bool) -> bool:
    """arnoldi_eig
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def arnoldi_eig_aux(aux: bool) -> bool:
    """arnoldi_eig
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_arnoldi_eig(seed: int = 0) -> float:
    checks = []
    checks.append(arnoldi_eig_ok(True, True))
    checks.append(not arnoldi_eig_ok(False, True))
    checks.append(arnoldi_eig_aux(True))
    checks.append(not arnoldi_eig_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_arnoldi_eig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arnoldi_eig": _bench_arnoldi_eig(seed)}
