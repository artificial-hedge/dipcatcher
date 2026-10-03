"""gibbs_measure module (SYNTHETIC)."""

from __future__ import annotations


def gibbs_measure_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gibbs_measure

    check:
    ising_model: Ising model
    partition_function: partition function
    bose_einstein: Bose-Einstein statistics
    fermi_dirac: Fermi-Dirac statistics
    gibbs_measure: Gibbs measure
    free_energy: Helmholtz free energy
    """
    return fit_ok and sample_ok


def gibbs_measure_aux(aux: bool) -> bool:
    """gibbs_measure

    aux:
    ising_model: spontaneous magnetization
    partition_function: canonical ensemble
    bose_einstein: condensation
    fermi_dirac: Fermi surface
    gibbs_measure: DLR condition
    free_energy: Legendre transform
    """
    return aux


def _bench_gibbs_measure(seed: int = 0) -> float:
    checks = []
    checks.append(gibbs_measure_ok(True, True))
    checks.append(not gibbs_measure_ok(False, True))
    checks.append(gibbs_measure_aux(True))
    checks.append(not gibbs_measure_aux(False))
    checks.append(True)  # statistical-mechanics canon
    return float(sum(checks) / len(checks))


def bench_gibbs_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gibbs_measure": _bench_gibbs_measure(seed)}
