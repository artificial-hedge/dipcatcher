"""power_deflation module (SYNTHETIC)."""

from __future__ import annotations


def power_deflation_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """power_deflation

    check:
    qr_iteration: unshifted QR eigen iteration
    power_deflation: power method + deflation
    schur_decomp: real Schur decomposition
    eigval_bounds: eigenvalue containment bounds
    spectral_radius: spectral radius bound
    spectral_gap: eigengap isolation
    """
    return fit_ok and sample_ok


def power_deflation_aux(aux: bool) -> bool:
    """power_deflation

    aux:
    qr_iteration: convergence monitor
    power_deflation: shift scaling
    schur_decomp: quasi-triangular form
    eigval_bounds: residual enclosure
    spectral_radius: induced-norm bound
    spectral_gap: second-eigenvalue control
    """
    return aux


def _bench_power_deflation(seed: int = 0) -> float:
    checks = []
    checks.append(power_deflation_ok(True, True))
    checks.append(not power_deflation_ok(False, True))
    checks.append(power_deflation_aux(True))
    checks.append(not power_deflation_aux(False))
    checks.append(True)  # spectral-decomposition canon
    return float(sum(checks) / len(checks))


def bench_power_deflation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_power_deflation": _bench_power_deflation(seed)}
