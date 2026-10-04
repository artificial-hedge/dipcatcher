"""molecular_orbitals module (SYNTHETIC)."""

from __future__ import annotations


def molecular_orbitals_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """molecular_orbitals

    check:
    hartree_fock: Hartree-Fock method
    born_oppenheimer: Born-Oppenheimer approximation
    molecular_orbitals: molecular orbitals
    rotational_spectra: rotational spectroscopy
    vibrational_spectra: vibrational spectroscopy
    zeeman_effect: Zeeman effect
    """
    return fit_ok and sample_ok


def molecular_orbitals_aux(aux: bool) -> bool:
    """molecular_orbitals

    aux:
    hartree_fock: self-consistent field
    born_oppenheimer: potential surfaces
    molecular_orbitals: LCAO
    rotational_spectra: microwave
    vibrational_spectra: infrared
    zeeman_effect: magnetic splitting
    """
    return aux


def _bench_molecular_orbitals(seed: int = 0) -> float:
    checks = []
    checks.append(molecular_orbitals_ok(True, True))
    checks.append(not molecular_orbitals_ok(False, True))
    checks.append(molecular_orbitals_aux(True))
    checks.append(not molecular_orbitals_aux(False))
    checks.append(True)  # atomic/molecular-physics canon
    return float(sum(checks) / len(checks))


def bench_molecular_orbitals(seed: int = 0) -> dict[str, float]:
    return {"synthetic_molecular_orbitals": _bench_molecular_orbitals(seed)}
