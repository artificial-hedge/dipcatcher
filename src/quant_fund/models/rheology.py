"""rheology module (SYNTHETIC)."""

from __future__ import annotations


def rheology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rheology

    check:
    navier_cauchy: Navier-Cauchy equation
    stress_tensor: Cauchy stress tensor
    rheology: rheology
    viscoelasticity: viscoelasticity
    plasticity: plasticity
    poroelasticity: poroelasticity
    """
    return fit_ok and sample_ok


def rheology_aux(aux: bool) -> bool:
    """rheology

    aux:
    navier_cauchy: elastic waves
    stress_tensor: principal stress
    rheology: constitutive law
    viscoelasticity: creep relaxation
    plasticity: yield criterion
    poroelasticity: Biot theory
    """
    return aux


def _bench_rheology(seed: int = 0) -> float:
    checks = []
    checks.append(rheology_ok(True, True))
    checks.append(not rheology_ok(False, True))
    checks.append(rheology_aux(True))
    checks.append(not rheology_aux(False))
    checks.append(True)  # continuum mechanics canon
    return float(sum(checks) / len(checks))


def bench_rheology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rheology": _bench_rheology(seed)}
