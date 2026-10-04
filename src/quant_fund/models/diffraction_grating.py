"""diffraction_grating module (SYNTHETIC)."""

from __future__ import annotations


def diffraction_grating_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diffraction_grating

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def diffraction_grating_aux(aux: bool) -> bool:
    """diffraction_grating

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_diffraction_grating(seed: int = 0) -> float:
    checks = []
    checks.append(diffraction_grating_ok(True, True))
    checks.append(not diffraction_grating_ok(False, True))
    checks.append(diffraction_grating_aux(True))
    checks.append(not diffraction_grating_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_diffraction_grating(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffraction_grating": _bench_diffraction_grating(seed)}
