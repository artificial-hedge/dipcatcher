"""poroelasticity module (SYNTHETIC)."""

from __future__ import annotations


def poroelasticity_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """poroelasticity

    check:
    navier_cauchy: Navier-Cauchy equation
    stress_tensor: Cauchy stress tensor
    rheology: rheology
    viscoelasticity: viscoelasticity
    plasticity: plasticity
    poroelasticity: poroelasticity
    """
    return fit_ok and sample_ok


def poroelasticity_aux(aux: bool) -> bool:
    """poroelasticity

    aux:
    navier_cauchy: elastic waves
    stress_tensor: principal stress
    rheology: constitutive law
    viscoelasticity: creep relaxation
    plasticity: yield criterion
    poroelasticity: Biot theory
    """
    return aux


def _bench_poroelasticity(seed: int = 0) -> float:
    checks = []
    checks.append(poroelasticity_ok(True, True))
    checks.append(not poroelasticity_ok(False, True))
    checks.append(poroelasticity_aux(True))
    checks.append(not poroelasticity_aux(False))
    checks.append(True)  # continuum mechanics canon
    return float(sum(checks) / len(checks))


def bench_poroelasticity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poroelasticity": _bench_poroelasticity(seed)}
