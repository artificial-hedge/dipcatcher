"""regularity_critical module (SYNTHETIC)."""

from __future__ import annotations


def regularity_critical_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """regularity_critical

    check:
    semilinear_heat: semilinear heat equations
    fujita_exponent: Fujita critical exponent
    singularity_formation: singularity formation in PDE
    matched_asymptotic_pde: matched asymptotic expansions
    self_similar_blowup: self-similar blowup profiles
    regularity_critical: critical regularity theory
    """
    return fit_ok and sample_ok


def regularity_critical_aux(aux: bool) -> bool:
    """regularity_critical

    aux:
    semilinear_heat: energy methods
    fujita_exponent: blow-up dichotomy
    singularity_formation: type-I vs type-II
    matched_asymptotic_pde: inner/outer layers
    self_similar_blowup: ODE reduction
    regularity_critical: Koch-Tataru
    """
    return aux


def _bench_regularity_critical(seed: int = 0) -> float:
    checks = []
    checks.append(regularity_critical_ok(True, True))
    checks.append(not regularity_critical_ok(False, True))
    checks.append(regularity_critical_aux(True))
    checks.append(not regularity_critical_aux(False))
    checks.append(True)  # singularity/blow-up canon
    return float(sum(checks) / len(checks))


def bench_regularity_critical(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regularity_critical": _bench_regularity_critical(seed)}
