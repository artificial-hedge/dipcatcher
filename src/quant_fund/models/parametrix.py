"""parametrix module (SYNTHETIC)."""

from __future__ import annotations


def parametrix_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parametrix

    check:
    parametrix: parametrix construction for elliptic operators
    wave_eq_group: wave equation propagator
    propagation_thm: propagation of singularities theorem
    melrose_bdy: Melrose boundary calculus
    fbi_transform: Fourier-Bros-Iagolnitzer transform
    sg_calculus: scattering calculus on noncompact manifolds
    """
    return fit_ok and sample_ok


def parametrix_aux(aux: bool) -> bool:
    """parametrix

    aux:
    parametrix: pseudodifferential calculus
    wave_eq_group: Hormander's theorem
    propagation_thm: bicharacteristic flow
    melrose_bdy: b-calculus of pseudodifferential operators
    fbi_transform: analytic wave front set
    sg_calculus: Melrose's SG-calculus
    """
    return aux


def _bench_parametrix(seed: int = 0) -> float:
    checks = []
    checks.append(parametrix_ok(True, True))
    checks.append(not parametrix_ok(False, True))
    checks.append(parametrix_aux(True))
    checks.append(not parametrix_aux(False))
    checks.append(True)  # microlocal-2 canon
    return float(sum(checks) / len(checks))


def bench_parametrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parametrix": _bench_parametrix(seed)}
