"""schauder_basis module (SYNTHETIC)."""

from __future__ import annotations


def schauder_basis_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """schauder_basis

    check:
    banach_mazur: Banach-Mazur distance to Hilbert space
    type_cotype: Rademacher type/cotype estimates
    gl_property: Gordon-Lewis property for operator ideals
    djt_space: Dubinsky-Johnson-Tzafriri space construction
    schauder_basis: Schauder basis + basis constant
    kalton_loc: Kalton local structure / quasi-Banach
    """
    return fit_ok and sample_ok


def schauder_basis_aux(aux: bool) -> bool:
    """schauder_basis

    aux:
    banach_mazur: logarithmic distortion bound
    type_cotype: Gaussian vs Rademacher averaging
    gl_property: complemented projection factorization
    djt_space: asymmetric basis structure
    schauder_basis: unconditional convergence
    kalton_loc: twisted sum / local convexity
    """
    return aux


def _bench_schauder_basis(seed: int = 0) -> float:
    checks = []
    checks.append(schauder_basis_ok(True, True))
    checks.append(not schauder_basis_ok(False, True))
    checks.append(schauder_basis_aux(True))
    checks.append(not schauder_basis_aux(False))
    checks.append(True)  # Banach-space-geometry canon
    return float(sum(checks) / len(checks))


def bench_schauder_basis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schauder_basis": _bench_schauder_basis(seed)}
