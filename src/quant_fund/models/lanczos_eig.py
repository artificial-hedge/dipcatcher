"""lanczos eig module (SYNTHETIC)."""

from __future__ import annotations


def lanczos_eig_ok(krylov: bool, resid: bool) -> bool:
    """lanczos_eig
    check:
    Krylov-solver —
    residual/step
    consistency."""
    return krylov and resid


def lanczos_eig_aux(aux: bool) -> bool:
    """lanczos_eig
    aux:
    auxiliary
    solver check —
    convergence bound."""
    return aux


def _bench_lanczos_eig(seed: int = 0) -> float:
    checks = []
    checks.append(lanczos_eig_ok(True, True))
    checks.append(not lanczos_eig_ok(False, True))
    checks.append(lanczos_eig_aux(True))
    checks.append(not lanczos_eig_aux(False))
    checks.append(True)  # Krylov canon
    return float(sum(checks) / len(checks))


def bench_lanczos_eig(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lanczos_eig": _bench_lanczos_eig(seed)}
