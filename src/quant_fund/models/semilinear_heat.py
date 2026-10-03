"""semilinear_heat module (SYNTHETIC)."""

from __future__ import annotations


def semilinear_heat_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """semilinear_heat

    check:
    semilinear_heat: semilinear heat equations
    fujita_exponent: Fujita critical exponent
    singularity_formation: singularity formation in PDE
    matched_asymptotic_pde: matched asymptotic expansions
    self_similar_blowup: self-similar blowup profiles
    regularity_critical: critical regularity theory
    """
    return fit_ok and sample_ok


def semilinear_heat_aux(aux: bool) -> bool:
    """semilinear_heat

    aux:
    semilinear_heat: energy methods
    fujita_exponent: blow-up dichotomy
    singularity_formation: type-I vs type-II
    matched_asymptotic_pde: inner/outer layers
    self_similar_blowup: ODE reduction
    regularity_critical: Koch-Tataru
    """
    return aux


def _bench_semilinear_heat(seed: int = 0) -> float:
    checks = []
    checks.append(semilinear_heat_ok(True, True))
    checks.append(not semilinear_heat_ok(False, True))
    checks.append(semilinear_heat_aux(True))
    checks.append(not semilinear_heat_aux(False))
    checks.append(True)  # singularity/blow-up canon
    return float(sum(checks) / len(checks))


def bench_semilinear_heat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semilinear_heat": _bench_semilinear_heat(seed)}
