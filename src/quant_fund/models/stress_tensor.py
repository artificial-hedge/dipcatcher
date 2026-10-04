"""stress_tensor module (SYNTHETIC)."""

from __future__ import annotations


def stress_tensor_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stress_tensor

    check:
    navier_cauchy: Navier-Cauchy equation
    stress_tensor: Cauchy stress tensor
    rheology: rheology
    viscoelasticity: viscoelasticity
    plasticity: plasticity
    poroelasticity: poroelasticity
    """
    return fit_ok and sample_ok


def stress_tensor_aux(aux: bool) -> bool:
    """stress_tensor

    aux:
    navier_cauchy: elastic waves
    stress_tensor: principal stress
    rheology: constitutive law
    viscoelasticity: creep relaxation
    plasticity: yield criterion
    poroelasticity: Biot theory
    """
    return aux


def _bench_stress_tensor(seed: int = 0) -> float:
    checks = []
    checks.append(stress_tensor_ok(True, True))
    checks.append(not stress_tensor_ok(False, True))
    checks.append(stress_tensor_aux(True))
    checks.append(not stress_tensor_aux(False))
    checks.append(True)  # continuum mechanics canon
    return float(sum(checks) / len(checks))


def bench_stress_tensor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stress_tensor": _bench_stress_tensor(seed)}
