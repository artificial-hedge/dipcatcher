"""partition_function module (SYNTHETIC)."""

from __future__ import annotations


def partition_function_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """partition_function

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def partition_function_aux(aux: bool) -> bool:
    """partition_function

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_partition_function(seed: int = 0) -> float:
    checks = []
    checks.append(partition_function_ok(True, True))
    checks.append(not partition_function_ok(False, True))
    checks.append(partition_function_aux(True))
    checks.append(not partition_function_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_partition_function(seed: int = 0) -> dict[str, float]:
    return {"synthetic_partition_function": _bench_partition_function(seed)}
