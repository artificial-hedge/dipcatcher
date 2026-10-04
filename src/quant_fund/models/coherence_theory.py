"""coherence_theory module (SYNTHETIC)."""

from __future__ import annotations


def coherence_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coherence_theory

    check:
    diffraction_grating: diffraction grating
    fourier_optics: Fourier optics
    interference_fringes: interference fringes
    polarization_states: polarization states
    coherence_theory: coherence theory
    holography: holography
    """
    return fit_ok and sample_ok


def coherence_theory_aux(aux: bool) -> bool:
    """coherence_theory

    aux:
    diffraction_grating: grating equation
    fourier_optics: spatial frequency
    interference_fringes: fringe visibility
    polarization_states: Jones calculus
    coherence_theory: mutual coherence
    holography: wavefront reconstruction
    """
    return aux


def _bench_coherence_theory(seed: int = 0) -> float:
    checks = []
    checks.append(coherence_theory_ok(True, True))
    checks.append(not coherence_theory_ok(False, True))
    checks.append(coherence_theory_aux(True))
    checks.append(not coherence_theory_aux(False))
    checks.append(True)  # optics-2 canon
    return float(sum(checks) / len(checks))


def bench_coherence_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coherence_theory": _bench_coherence_theory(seed)}
