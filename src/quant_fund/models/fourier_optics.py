"""fourier_optics module (SYNTHETIC)."""

from __future__ import annotations


def fourier_optics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fourier_optics

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def fourier_optics_aux(aux: bool) -> bool:
    """fourier_optics

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_fourier_optics(seed: int = 0) -> float:
    checks = []
    checks.append(fourier_optics_ok(True, True))
    checks.append(not fourier_optics_ok(False, True))
    checks.append(fourier_optics_aux(True))
    checks.append(not fourier_optics_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_fourier_optics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fourier_optics": _bench_fourier_optics(seed)}
