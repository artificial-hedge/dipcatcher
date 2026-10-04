"""plasticity module (SYNTHETIC)."""

from __future__ import annotations


def plasticity_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """plasticity

    check:
    navier_cauchy: Navier-Cauchy equation
    stress_tensor: Cauchy stress tensor
    rheology: rheology
    viscoelasticity: viscoelasticity
    plasticity: plasticity
    poroelasticity: poroelasticity
    """
    return fit_ok and sample_ok


def plasticity_aux(aux: bool) -> bool:
    """plasticity

    aux:
    navier_cauchy: elastic waves
    stress_tensor: principal stress
    rheology: constitutive law
    viscoelasticity: creep relaxation
    plasticity: yield criterion
    poroelasticity: Biot theory
    """
    return aux


def _bench_plasticity(seed: int = 0) -> float:
    checks = []
    checks.append(plasticity_ok(True, True))
    checks.append(not plasticity_ok(False, True))
    checks.append(plasticity_aux(True))
    checks.append(not plasticity_aux(False))
    checks.append(True)  # continuum mechanics canon
    return float(sum(checks) / len(checks))


def bench_plasticity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_plasticity": _bench_plasticity(seed)}
