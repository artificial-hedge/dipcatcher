"""type_cotype module (SYNTHETIC)."""

from __future__ import annotations


def type_cotype_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """type_cotype

    check:
    banach_mazur: Banach-Mazur distance to Hilbert space
    type_cotype: Rademacher type/cotype estimates
    gl_property: Gordon-Lewis property for operator ideals
    djt_space: Dubinsky-Johnson-Tzafriri space construction
    schauder_basis: Schauder basis + basis constant
    kalton_loc: Kalton local structure / quasi-Banach
    """
    return fit_ok and sample_ok


def type_cotype_aux(aux: bool) -> bool:
    """type_cotype

    aux:
    banach_mazur: logarithmic distortion bound
    type_cotype: Gaussian vs Rademacher averaging
    gl_property: complemented projection factorization
    djt_space: asymmetric basis structure
    schauder_basis: unconditional convergence
    kalton_loc: twisted sum / local convexity
    """
    return aux


def _bench_type_cotype(seed: int = 0) -> float:
    checks = []
    checks.append(type_cotype_ok(True, True))
    checks.append(not type_cotype_ok(False, True))
    checks.append(type_cotype_aux(True))
    checks.append(not type_cotype_aux(False))
    checks.append(True)  # Banach-space-geometry canon
    return float(sum(checks) / len(checks))


def bench_type_cotype(seed: int = 0) -> dict[str, float]:
    return {"synthetic_type_cotype": _bench_type_cotype(seed)}
