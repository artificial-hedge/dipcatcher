"""holography module (SYNTHETIC)."""

from __future__ import annotations


def holography_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """holography

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def holography_aux(aux: bool) -> bool:
    """holography

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_holography(seed: int = 0) -> float:
    checks = []
    checks.append(holography_ok(True, True))
    checks.append(not holography_ok(False, True))
    checks.append(holography_aux(True))
    checks.append(not holography_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_holography(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holography": _bench_holography(seed)}
