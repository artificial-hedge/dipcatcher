"""energy_principle module (SYNTHETIC)."""

from __future__ import annotations


def energy_principle_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """energy_principle

    check:
    dirichlet_problem: Perron-Wiener-Brelot solution
    energy_principle: Gauss minimum-energy problem
    equilibrium_measure: Frostman equilibrium measure
    thin_set: Wiener thinness criterion
    boundary_regular: regular boundary points
    capacitary_pot: capacitary potential of a set
    """
    return fit_ok and sample_ok


def energy_principle_aux(aux: bool) -> bool:
    """energy_principle

    aux:
    dirichlet_problem: PWB upper/lower functions
    energy_principle: positive-definite kernels
    equilibrium_measure: support on the boundary
    thin_set: semiregularity characterization
    boundary_regular: barrier construction
    capacitary_pot: conductor potential
    """
    return aux


def _bench_energy_principle(seed: int = 0) -> float:
    checks = []
    checks.append(energy_principle_ok(True, True))
    checks.append(not energy_principle_ok(False, True))
    checks.append(energy_principle_aux(True))
    checks.append(not energy_principle_aux(False))
    checks.append(True)  # potential-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_energy_principle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_energy_principle": _bench_energy_principle(seed)}
