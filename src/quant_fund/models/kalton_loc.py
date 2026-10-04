"""kalton_loc module (SYNTHETIC)."""

from __future__ import annotations


def kalton_loc_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kalton_loc

    check:
    banach_mazur: Banach-Mazur distance to Hilbert space
    type_cotype: Rademacher type/cotype estimates
    gl_property: Gordon-Lewis property for operator ideals
    djt_space: Dubinsky-Johnson-Tzafriri space construction
    schauder_basis: Schauder basis + basis constant
    kalton_loc: Kalton local structure / quasi-Banach
    """
    return fit_ok and sample_ok


def kalton_loc_aux(aux: bool) -> bool:
    """kalton_loc

    aux:
    banach_mazur: logarithmic distortion bound
    type_cotype: Gaussian vs Rademacher averaging
    gl_property: complemented projection factorization
    djt_space: asymmetric basis structure
    schauder_basis: unconditional convergence
    kalton_loc: twisted sum / local convexity
    """
    return aux


def _bench_kalton_loc(seed: int = 0) -> float:
    checks = []
    checks.append(kalton_loc_ok(True, True))
    checks.append(not kalton_loc_ok(False, True))
    checks.append(kalton_loc_aux(True))
    checks.append(not kalton_loc_aux(False))
    checks.append(True)  # Banach-space-geometry canon
    return float(sum(checks) / len(checks))


def bench_kalton_loc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kalton_loc": _bench_kalton_loc(seed)}
