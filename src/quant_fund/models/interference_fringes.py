"""interference_fringes module (SYNTHETIC)."""

from __future__ import annotations


def interference_fringes_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interference_fringes

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def interference_fringes_aux(aux: bool) -> bool:
    """interference_fringes

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_interference_fringes(seed: int = 0) -> float:
    checks = []
    checks.append(interference_fringes_ok(True, True))
    checks.append(not interference_fringes_ok(False, True))
    checks.append(interference_fringes_aux(True))
    checks.append(not interference_fringes_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_interference_fringes(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interference_fringes": _bench_interference_fringes(seed)}
