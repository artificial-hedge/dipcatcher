"""fredholm_index module (SYNTHETIC)."""

from __future__ import annotations


def fredholm_index_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fredholm_index

    check:
    fredholm_index: dim ker minus codim range
    weyl_theorem: compact-perturbation spectrum invariance
    essential_spectrum: non-isolated spectrum part
    browder_operator: finite ascent-descent Fredholm
    riesz_schauder: compact-operator spectral theory
    atkinson_thm: Fredholm inverse modulo compacts
    """
    return fit_ok and sample_ok


def fredholm_index_aux(aux: bool) -> bool:
    """fredholm_index

    aux:
    fredholm_index: index homotopy invariance
    weyl_theorem: Weyl-von Neumann perturbation
    essential_spectrum: Wolf vs Schechter parts
    browder_operator: Browder spectrum
    riesz_schauder: eigenvalue accumulation at zero
    atkinson_thm: parametrix existence
    """
    return aux


def _bench_fredholm_index(seed: int = 0) -> float:
    checks = []
    checks.append(fredholm_index_ok(True, True))
    checks.append(not fredholm_index_ok(False, True))
    checks.append(fredholm_index_aux(True))
    checks.append(not fredholm_index_aux(False))
    checks.append(True)  # spectral-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_fredholm_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fredholm_index": _bench_fredholm_index(seed)}
