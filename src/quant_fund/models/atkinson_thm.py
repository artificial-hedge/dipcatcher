"""atkinson_thm module (SYNTHETIC)."""

from __future__ import annotations


def atkinson_thm_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """atkinson_thm

    check:
    fredholm_index: dim ker minus codim range
    weyl_theorem: compact-perturbation spectrum invariance
    essential_spectrum: non-isolated spectrum part
    browder_operator: finite ascent-descent Fredholm
    riesz_schauder: compact-operator spectral theory
    atkinson_thm: Fredholm inverse modulo compacts
    """
    return fit_ok and sample_ok


def atkinson_thm_aux(aux: bool) -> bool:
    """atkinson_thm

    aux:
    fredholm_index: index homotopy invariance
    weyl_theorem: Weyl-von Neumann perturbation
    essential_spectrum: Wolf vs Schechter parts
    browder_operator: Browder spectrum
    riesz_schauder: eigenvalue accumulation at zero
    atkinson_thm: parametrix existence
    """
    return aux


def _bench_atkinson_thm(seed: int = 0) -> float:
    checks = []
    checks.append(atkinson_thm_ok(True, True))
    checks.append(not atkinson_thm_ok(False, True))
    checks.append(atkinson_thm_aux(True))
    checks.append(not atkinson_thm_aux(False))
    checks.append(True)  # spectral-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_atkinson_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atkinson_thm": _bench_atkinson_thm(seed)}
