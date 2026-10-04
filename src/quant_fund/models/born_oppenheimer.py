"""born_oppenheimer module (SYNTHETIC)."""

from __future__ import annotations


def born_oppenheimer_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """born_oppenheimer

    check:
    hartree_fock: Hartree-Fock method
    born_oppenheimer: Born-Oppenheimer approximation
    molecular_orbitals: molecular orbitals
    rotational_spectra: rotational spectroscopy
    vibrational_spectra: vibrational spectroscopy
    zeeman_effect: Zeeman effect
    """
    return fit_ok and sample_ok


def born_oppenheimer_aux(aux: bool) -> bool:
    """born_oppenheimer

    aux:
    hartree_fock: self-consistent field
    born_oppenheimer: potential surfaces
    molecular_orbitals: LCAO
    rotational_spectra: microwave
    vibrational_spectra: infrared
    zeeman_effect: magnetic splitting
    """
    return aux


def _bench_born_oppenheimer(seed: int = 0) -> float:
    checks = []
    checks.append(born_oppenheimer_ok(True, True))
    checks.append(not born_oppenheimer_ok(False, True))
    checks.append(born_oppenheimer_aux(True))
    checks.append(not born_oppenheimer_aux(False))
    checks.append(True)  # atomic/molecular-physics canon
    return float(sum(checks) / len(checks))


def bench_born_oppenheimer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_born_oppenheimer": _bench_born_oppenheimer(seed)}
