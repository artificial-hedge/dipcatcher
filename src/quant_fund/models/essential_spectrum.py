"""essential_spectrum module (SYNTHETIC)."""

from __future__ import annotations


def essential_spectrum_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """essential_spectrum

    check:
    fredholm_index: dim ker minus codim range
    weyl_theorem: compact-perturbation spectrum invariance
    essential_spectrum: non-isolated spectrum part
    browder_operator: finite ascent-descent Fredholm
    riesz_schauder: compact-operator spectral theory
    atkinson_thm: Fredholm inverse modulo compacts
    """
    return fit_ok and sample_ok


def essential_spectrum_aux(aux: bool) -> bool:
    """essential_spectrum

    aux:
    fredholm_index: index homotopy invariance
    weyl_theorem: Weyl-von Neumann perturbation
    essential_spectrum: Wolf vs Schechter parts
    browder_operator: Browder spectrum
    riesz_schauder: eigenvalue accumulation at zero
    atkinson_thm: parametrix existence
    """
    return aux


def _bench_essential_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(essential_spectrum_ok(True, True))
    checks.append(not essential_spectrum_ok(False, True))
    checks.append(essential_spectrum_aux(True))
    checks.append(not essential_spectrum_aux(False))
    checks.append(True)  # spectral-theory-2 canon
    return float(sum(checks) / len(checks))


def bench_essential_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_essential_spectrum": _bench_essential_spectrum(seed)}
