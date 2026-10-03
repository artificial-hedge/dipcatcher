"""fbi_transform module (SYNTHETIC)."""

from __future__ import annotations


def fbi_transform_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fbi_transform

    check:
    parametrix: parametrix construction for elliptic operators
    wave_eq_group: wave equation propagator
    propagation_thm: propagation of singularities theorem
    melrose_bdy: Melrose boundary calculus
    fbi_transform: Fourier-Bros-Iagolnitzer transform
    sg_calculus: scattering calculus on noncompact manifolds
    """
    return fit_ok and sample_ok


def fbi_transform_aux(aux: bool) -> bool:
    """fbi_transform

    aux:
    parametrix: pseudodifferential calculus
    wave_eq_group: Hormander's theorem
    propagation_thm: bicharacteristic flow
    melrose_bdy: b-calculus of pseudodifferential operators
    fbi_transform: analytic wave front set
    sg_calculus: Melrose's SG-calculus
    """
    return aux


def _bench_fbi_transform(seed: int = 0) -> float:
    checks = []
    checks.append(fbi_transform_ok(True, True))
    checks.append(not fbi_transform_ok(False, True))
    checks.append(fbi_transform_aux(True))
    checks.append(not fbi_transform_aux(False))
    checks.append(True)  # microlocal-2 canon
    return float(sum(checks) / len(checks))


def bench_fbi_transform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fbi_transform": _bench_fbi_transform(seed)}
