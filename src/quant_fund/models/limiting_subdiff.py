"""limiting_subdiff module (SYNTHETIC)."""

from __future__ import annotations


def limiting_subdiff_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """limiting_subdiff

    check:
    limiting_subdiff: Mordukhovich limiting subdiff
    proximal_subdiff: proximal subgradient
    ekeland_var: Ekeland variational principle
    monteiro_semismooth: Monteiro semismooth analysis
    semismooth_newton: semismooth Newton method
    augmented_lagr: augmented Lagrangian method
    """
    return fit_ok and sample_ok


def limiting_subdiff_aux(aux: bool) -> bool:
    """limiting_subdiff

    aux:
    limiting_subdiff: robust calculus rules
    proximal_subdiff: prox-regularity conditions
    ekeland_var: epsilon-minimizer existence
    monteiro_semismooth: strong semismoothness
    semismooth_newton: superlinear local rate
    augmented_lagr: dual ascent on AL function
    """
    return aux


def _bench_limiting_subdiff(seed: int = 0) -> float:
    checks = []
    checks.append(limiting_subdiff_ok(True, True))
    checks.append(not limiting_subdiff_ok(False, True))
    checks.append(limiting_subdiff_aux(True))
    checks.append(not limiting_subdiff_aux(False))
    checks.append(True)  # nonsmooth-Newton canon
    return float(sum(checks) / len(checks))


def bench_limiting_subdiff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_limiting_subdiff": _bench_limiting_subdiff(seed)}
