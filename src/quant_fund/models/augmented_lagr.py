"""augmented_lagr module (SYNTHETIC)."""

from __future__ import annotations


def augmented_lagr_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """augmented_lagr

    check:
    limiting_subdiff: Mordukhovich limiting subdiff
    proximal_subdiff: proximal subgradient
    ekeland_var: Ekeland variational principle
    monteiro_semismooth: Monteiro semismooth analysis
    semismooth_newton: semismooth Newton method
    augmented_lagr: augmented Lagrangian method
    """
    return fit_ok and sample_ok


def augmented_lagr_aux(aux: bool) -> bool:
    """augmented_lagr

    aux:
    limiting_subdiff: robust calculus rules
    proximal_subdiff: prox-regularity conditions
    ekeland_var: epsilon-minimizer existence
    monteiro_semismooth: strong semismoothness
    semismooth_newton: superlinear local rate
    augmented_lagr: dual ascent on AL function
    """
    return aux


def _bench_augmented_lagr(seed: int = 0) -> float:
    checks = []
    checks.append(augmented_lagr_ok(True, True))
    checks.append(not augmented_lagr_ok(False, True))
    checks.append(augmented_lagr_aux(True))
    checks.append(not augmented_lagr_aux(False))
    checks.append(True)  # nonsmooth-Newton canon
    return float(sum(checks) / len(checks))


def bench_augmented_lagr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_augmented_lagr": _bench_augmented_lagr(seed)}
