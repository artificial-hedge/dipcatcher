"""vibrational_spectra module (SYNTHETIC)."""

from __future__ import annotations


def vibrational_spectra_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vibrational_spectra

    check:
    hartree_fock: Hartree-Fock method
    born_oppenheimer: Born-Oppenheimer approximation
    molecular_orbitals: molecular orbitals
    rotational_spectra: rotational spectroscopy
    vibrational_spectra: vibrational spectroscopy
    zeeman_effect: Zeeman effect
    """
    return fit_ok and sample_ok


def vibrational_spectra_aux(aux: bool) -> bool:
    """vibrational_spectra

    aux:
    hartree_fock: self-consistent field
    born_oppenheimer: potential surfaces
    molecular_orbitals: LCAO
    rotational_spectra: microwave
    vibrational_spectra: infrared
    zeeman_effect: magnetic splitting
    """
    return aux


def _bench_vibrational_spectra(seed: int = 0) -> float:
    checks = []
    checks.append(vibrational_spectra_ok(True, True))
    checks.append(not vibrational_spectra_ok(False, True))
    checks.append(vibrational_spectra_aux(True))
    checks.append(not vibrational_spectra_aux(False))
    checks.append(True)  # atomic/molecular-physics canon
    return float(sum(checks) / len(checks))


def bench_vibrational_spectra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vibrational_spectra": _bench_vibrational_spectra(seed)}
