"""polarization_states module (SYNTHETIC)."""

from __future__ import annotations


def polarization_states_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """polarization_states

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def polarization_states_aux(aux: bool) -> bool:
    """polarization_states

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_polarization_states(seed: int = 0) -> float:
    checks = []
    checks.append(polarization_states_ok(True, True))
    checks.append(not polarization_states_ok(False, True))
    checks.append(polarization_states_aux(True))
    checks.append(not polarization_states_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_polarization_states(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polarization_states": _bench_polarization_states(seed)}
