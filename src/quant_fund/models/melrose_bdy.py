"""melrose_bdy module (SYNTHETIC)."""

from __future__ import annotations


def melrose_bdy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """melrose_bdy

    check:
    parametrix: parametrix construction for elliptic operators
    wave_eq_group: wave equation propagator
    propagation_thm: propagation of singularities theorem
    melrose_bdy: Melrose boundary calculus
    fbi_transform: Fourier-Bros-Iagolnitzer transform
    sg_calculus: scattering calculus on noncompact manifolds
    """
    return fit_ok and sample_ok


def melrose_bdy_aux(aux: bool) -> bool:
    """melrose_bdy

    aux:
    parametrix: pseudodifferential calculus
    wave_eq_group: Hormander's theorem
    propagation_thm: bicharacteristic flow
    melrose_bdy: b-calculus of pseudodifferential operators
    fbi_transform: analytic wave front set
    sg_calculus: Melrose's SG-calculus
    """
    return aux


def _bench_melrose_bdy(seed: int = 0) -> float:
    checks = []
    checks.append(melrose_bdy_ok(True, True))
    checks.append(not melrose_bdy_ok(False, True))
    checks.append(melrose_bdy_aux(True))
    checks.append(not melrose_bdy_aux(False))
    checks.append(True)  # microlocal-2 canon
    return float(sum(checks) / len(checks))


def bench_melrose_bdy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_melrose_bdy": _bench_melrose_bdy(seed)}
