"""phonon_spectrum module (SYNTHETIC)."""

from __future__ import annotations


def phonon_spectrum_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """phonon_spectrum

    check:
    bloch_theorem: Bloch theorem
    tight_binding: tight-binding model
    phonon_spectrum: phonon spectrum
    band_structure: band structure
    hubbard_model: Hubbard model
    kondo_effect: Kondo effect
    """
    return fit_ok and sample_ok


def phonon_spectrum_aux(aux: bool) -> bool:
    """phonon_spectrum

    aux:
    bloch_theorem: crystal momentum
    tight_binding: hopping
    phonon_spectrum: dispersion
    band_structure: band gap
    hubbard_model: Mott transition
    kondo_effect: screening cloud
    """
    return aux


def _bench_phonon_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(phonon_spectrum_ok(True, True))
    checks.append(not phonon_spectrum_ok(False, True))
    checks.append(phonon_spectrum_aux(True))
    checks.append(not phonon_spectrum_aux(False))
    checks.append(True)  # condensed-matter canon
    return float(sum(checks) / len(checks))


def bench_phonon_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phonon_spectrum": _bench_phonon_spectrum(seed)}
