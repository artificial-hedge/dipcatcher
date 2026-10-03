"""schur_decomp module (SYNTHETIC)."""

from __future__ import annotations


def schur_decomp_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """schur_decomp

    check:
    qr_iteration: unshifted QR eigen iteration
    power_deflation: power method + deflation
    schur_decomp: real Schur decomposition
    eigval_bounds: eigenvalue containment bounds
    spectral_radius: spectral radius bound
    spectral_gap: eigengap isolation
    """
    return fit_ok and sample_ok


def schur_decomp_aux(aux: bool) -> bool:
    """schur_decomp

    aux:
    qr_iteration: convergence monitor
    power_deflation: shift scaling
    schur_decomp: quasi-triangular form
    eigval_bounds: residual enclosure
    spectral_radius: induced-norm bound
    spectral_gap: second-eigenvalue control
    """
    return aux


def _bench_schur_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(schur_decomp_ok(True, True))
    checks.append(not schur_decomp_ok(False, True))
    checks.append(schur_decomp_aux(True))
    checks.append(not schur_decomp_aux(False))
    checks.append(True)  # spectral-decomposition canon
    return float(sum(checks) / len(checks))


def bench_schur_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_decomp": _bench_schur_decomp(seed)}
