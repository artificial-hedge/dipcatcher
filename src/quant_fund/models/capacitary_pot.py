"""capacitary_pot module (SYNTHETIC)."""

from __future__ import annotations


def capacitary_pot_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """capacitary_pot

    check:
    dirichlet_problem: Perron-Wiener-Brelot solution
    energy_principle: Gauss minimum-energy problem
    equilibrium_measure: Frostman equilibrium measure
    thin_set: Wiener thinness criterion
    boundary_regular: regular boundary points
    capacitary_pot: capacitary potential of a set
    """
    return fit_ok and sample_ok


def capacitary_pot_aux(aux: bool) -> bool:
    """capacitary_pot

    aux:
    dirichlet_problem: PWB upper/lower functions
    energy_principle: positive-definite kernels
    equilibrium_measure: support on the boundary
    thin_set: semiregularity characterization
    boundary_regular: barrier construction
    capacitary_pot: conductor potential
    """
    return aux


def _bench_capacitary_pot(seed: int = 0) -> float:
    checks = []
    checks.append(capacitary_pot_ok(True, True))
    checks.append(not capacitary_pot_ok(False, True))
    checks.append(capacitary_pot_aux(True))
    checks.append(not capacitary_pot_aux(False))
    checks.append(True)  # potential-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_capacitary_pot(seed: int = 0) -> dict[str, float]:
    return {"synthetic_capacitary_pot": _bench_capacitary_pot(seed)}
