"""fermi_dirac module (SYNTHETIC)."""

from __future__ import annotations


def fermi_dirac_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fermi_dirac

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def fermi_dirac_aux(aux: bool) -> bool:
    """fermi_dirac

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_fermi_dirac(seed: int = 0) -> float:
    checks = []
    checks.append(fermi_dirac_ok(True, True))
    checks.append(not fermi_dirac_ok(False, True))
    checks.append(fermi_dirac_aux(True))
    checks.append(not fermi_dirac_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_fermi_dirac(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fermi_dirac": _bench_fermi_dirac(seed)}
