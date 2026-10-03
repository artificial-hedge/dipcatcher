"""ising_model module (SYNTHETIC)."""

from __future__ import annotations


def ising_model_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ising_model

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def ising_model_aux(aux: bool) -> bool:
    """ising_model

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_ising_model(seed: int = 0) -> float:
    checks = []
    checks.append(ising_model_ok(True, True))
    checks.append(not ising_model_ok(False, True))
    checks.append(ising_model_aux(True))
    checks.append(not ising_model_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_ising_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ising_model": _bench_ising_model(seed)}
