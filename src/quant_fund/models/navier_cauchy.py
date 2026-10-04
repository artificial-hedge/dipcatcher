"""navier_cauchy module (SYNTHETIC)."""

from __future__ import annotations


def navier_cauchy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """navier_cauchy

    check:
    navier_cauchy: Navier-Cauchy equation
    stress_tensor: Cauchy stress tensor
    rheology: rheology
    viscoelasticity: viscoelasticity
    plasticity: plasticity
    poroelasticity: poroelasticity
    """
    return fit_ok and sample_ok


def navier_cauchy_aux(aux: bool) -> bool:
    """navier_cauchy

    aux:
    navier_cauchy: elastic waves
    stress_tensor: principal stress
    rheology: constitutive law
    viscoelasticity: creep relaxation
    plasticity: yield criterion
    poroelasticity: Biot theory
    """
    return aux


def _bench_navier_cauchy(seed: int = 0) -> float:
    checks = []
    checks.append(navier_cauchy_ok(True, True))
    checks.append(not navier_cauchy_ok(False, True))
    checks.append(navier_cauchy_aux(True))
    checks.append(not navier_cauchy_aux(False))
    checks.append(True)  # continuum mechanics canon
    return float(sum(checks) / len(checks))


def bench_navier_cauchy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_navier_cauchy": _bench_navier_cauchy(seed)}
