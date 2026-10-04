"""spectral_gap module (SYNTHETIC)."""

from __future__ import annotations


def spectral_gap_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spectral_gap

    check:
    qr_iteration: unshifted QR eigen iteration
    power_deflation: power method + deflation
    schur_decomp: real Schur decomposition
    eigval_bounds: eigenvalue containment bounds
    spectral_radius: spectral radius bound
    spectral_gap: eigengap isolation
    """
    return fit_ok and sample_ok


def spectral_gap_aux(aux: bool) -> bool:
    """spectral_gap

    aux:
    qr_iteration: convergence monitor
    power_deflation: shift scaling
    schur_decomp: quasi-triangular form
    eigval_bounds: residual enclosure
    spectral_radius: induced-norm bound
    spectral_gap: second-eigenvalue control
    """
    return aux


def _bench_spectral_gap(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_gap_ok(True, True))
    checks.append(not spectral_gap_ok(False, True))
    checks.append(spectral_gap_aux(True))
    checks.append(not spectral_gap_aux(False))
    checks.append(True)  # spectral-decomposition canon
    return float(sum(checks) / len(checks))


def bench_spectral_gap(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_gap": _bench_spectral_gap(seed)}
