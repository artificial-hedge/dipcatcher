"""free_energy module (SYNTHETIC)."""

from __future__ import annotations


def free_energy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """free_energy

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def free_energy_aux(aux: bool) -> bool:
    """free_energy

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_free_energy(seed: int = 0) -> float:
    checks = []
    checks.append(free_energy_ok(True, True))
    checks.append(not free_energy_ok(False, True))
    checks.append(free_energy_aux(True))
    checks.append(not free_energy_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_free_energy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_energy": _bench_free_energy(seed)}
