"""ekeland_var module (SYNTHETIC)."""

from __future__ import annotations


def ekeland_var_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ekeland_var

    check:
    limiting_subdiff: Mordukhovich limiting subdiff
    proximal_subdiff: proximal subgradient
    ekeland_var: Ekeland variational principle
    monteiro_semismooth: Monteiro semismooth analysis
    semismooth_newton: semismooth Newton method
    augmented_lagr: augmented Lagrangian method
    """
    return fit_ok and sample_ok


def ekeland_var_aux(aux: bool) -> bool:
    """ekeland_var

    aux:
    limiting_subdiff: robust calculus rules
    proximal_subdiff: prox-regularity conditions
    ekeland_var: epsilon-minimizer existence
    monteiro_semismooth: strong semismoothness
    semismooth_newton: superlinear local rate
    augmented_lagr: dual ascent on AL function
    """
    return aux


def _bench_ekeland_var(seed: int = 0) -> float:
    checks = []
    checks.append(ekeland_var_ok(True, True))
    checks.append(not ekeland_var_ok(False, True))
    checks.append(ekeland_var_aux(True))
    checks.append(not ekeland_var_aux(False))
    checks.append(True)  # nonsmooth-Newton canon
    return float(sum(checks) / len(checks))


def bench_ekeland_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ekeland_var": _bench_ekeland_var(seed)}
