"""thin_set module (SYNTHETIC)."""

from __future__ import annotations


def thin_set_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thin_set

    check:
    dirichlet_problem: Perron-Wiener-Brelot solution
    energy_principle: Gauss minimum-energy problem
    equilibrium_measure: Frostman equilibrium measure
    thin_set: Wiener thinness criterion
    boundary_regular: regular boundary points
    capacitary_pot: capacitary potential of a set
    """
    return fit_ok and sample_ok


def thin_set_aux(aux: bool) -> bool:
    """thin_set

    aux:
    dirichlet_problem: PWB upper/lower functions
    energy_principle: positive-definite kernels
    equilibrium_measure: support on the boundary
    thin_set: semiregularity characterization
    boundary_regular: barrier construction
    capacitary_pot: conductor potential
    """
    return aux


def _bench_thin_set(seed: int = 0) -> float:
    checks = []
    checks.append(thin_set_ok(True, True))
    checks.append(not thin_set_ok(False, True))
    checks.append(thin_set_aux(True))
    checks.append(not thin_set_aux(False))
    checks.append(True)  # potential-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_thin_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thin_set": _bench_thin_set(seed)}
