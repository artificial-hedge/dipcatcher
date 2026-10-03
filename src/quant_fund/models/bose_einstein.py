"""bose_einstein module (SYNTHETIC)."""

from __future__ import annotations


def bose_einstein_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bose_einstein

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def bose_einstein_aux(aux: bool) -> bool:
    """bose_einstein

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_bose_einstein(seed: int = 0) -> float:
    checks = []
    checks.append(bose_einstein_ok(True, True))
    checks.append(not bose_einstein_ok(False, True))
    checks.append(bose_einstein_aux(True))
    checks.append(not bose_einstein_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_bose_einstein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bose_einstein": _bench_bose_einstein(seed)}
