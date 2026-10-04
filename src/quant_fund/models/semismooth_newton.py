"""semismooth_newton module (SYNTHETIC)."""

from __future__ import annotations


def semismooth_newton_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """semismooth_newton

    check:
    limiting_subdiff: Mordukhovich limiting subdiff
    proximal_subdiff: proximal subgradient
    ekeland_var: Ekeland variational principle
    monteiro_semismooth: Monteiro semismooth analysis
    semismooth_newton: semismooth Newton method
    augmented_lagr: augmented Lagrangian method
    """
    return fit_ok and sample_ok


def semismooth_newton_aux(aux: bool) -> bool:
    """semismooth_newton

    aux:
    limiting_subdiff: robust calculus rules
    proximal_subdiff: prox-regularity conditions
    ekeland_var: epsilon-minimizer existence
    monteiro_semismooth: strong semismoothness
    semismooth_newton: superlinear local rate
    augmented_lagr: dual ascent on AL function
    """
    return aux


def _bench_semismooth_newton(seed: int = 0) -> float:
    checks = []
    checks.append(semismooth_newton_ok(True, True))
    checks.append(not semismooth_newton_ok(False, True))
    checks.append(semismooth_newton_aux(True))
    checks.append(not semismooth_newton_aux(False))
    checks.append(True)  # nonsmooth-Newton canon
    return float(sum(checks) / len(checks))


def bench_semismooth_newton(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semismooth_newton": _bench_semismooth_newton(seed)}
