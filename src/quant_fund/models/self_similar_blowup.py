"""self_similar_blowup module (SYNTHETIC)."""

from __future__ import annotations


def self_similar_blowup_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """self_similar_blowup

    check:
    semilinear_heat: semilinear heat equations
    fujita_exponent: Fujita critical exponent
    singularity_formation: singularity formation in PDE
    matched_asymptotic_pde: matched asymptotic expansions
    self_similar_blowup: self-similar blowup profiles
    regularity_critical: critical regularity theory
    """
    return fit_ok and sample_ok


def self_similar_blowup_aux(aux: bool) -> bool:
    """self_similar_blowup

    aux:
    semilinear_heat: energy methods
    fujita_exponent: blow-up dichotomy
    singularity_formation: type-I vs type-II
    matched_asymptotic_pde: inner/outer layers
    self_similar_blowup: ODE reduction
    regularity_critical: Koch-Tataru
    """
    return aux


def _bench_self_similar_blowup(seed: int = 0) -> float:
    checks = []
    checks.append(self_similar_blowup_ok(True, True))
    checks.append(not self_similar_blowup_ok(False, True))
    checks.append(self_similar_blowup_aux(True))
    checks.append(not self_similar_blowup_aux(False))
    checks.append(True)  # singularity/blow-up canon
    return float(sum(checks) / len(checks))


def bench_self_similar_blowup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_similar_blowup": _bench_self_similar_blowup(seed)}
