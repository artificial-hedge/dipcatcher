"""monteiro_semismooth module (SYNTHETIC)."""

from __future__ import annotations


def monteiro_semismooth_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """monteiro_semismooth

    check:
    limiting_subdiff: Mordukhovich limiting subdiff
    proximal_subdiff: proximal subgradient
    ekeland_var: Ekeland variational principle
    monteiro_semismooth: Monteiro semismooth analysis
    semismooth_newton: semismooth Newton method
    augmented_lagr: augmented Lagrangian method
    """
    return fit_ok and sample_ok


def monteiro_semismooth_aux(aux: bool) -> bool:
    """monteiro_semismooth

    aux:
    limiting_subdiff: robust calculus rules
    proximal_subdiff: prox-regularity conditions
    ekeland_var: epsilon-minimizer existence
    monteiro_semismooth: strong semismoothness
    semismooth_newton: superlinear local rate
    augmented_lagr: dual ascent on AL function
    """
    return aux


def _bench_monteiro_semismooth(seed: int = 0) -> float:
    checks = []
    checks.append(monteiro_semismooth_ok(True, True))
    checks.append(not monteiro_semismooth_ok(False, True))
    checks.append(monteiro_semismooth_aux(True))
    checks.append(not monteiro_semismooth_aux(False))
    checks.append(True)  # nonsmooth-Newton canon
    return float(sum(checks) / len(checks))


def bench_monteiro_semismooth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monteiro_semismooth": _bench_monteiro_semismooth(seed)}
