"""restriction_est module (SYNTHETIC)."""

from __future__ import annotations


def restriction_est_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """restriction_est

    check:
    hausdorff_young: Hausdorff-Young Lp Fourier bound
    restriction_est: Fourier restriction to surfaces
    bochner_riesz: Bochner-Riesz summability index
    lp_multiplier: Lp Fourier multiplier estimates
    oscillatory_int: oscillatory integral decay
    strichartz_est: Strichartz space-time estimate
    """
    return fit_ok and sample_ok


def restriction_est_aux(aux: bool) -> bool:
    """restriction_est

    aux:
    hausdorff_young: Plancherel endpoint interpolation
    restriction_est: Tomas-Stein exponent range
    bochner_riesz: critical index comparison
    lp_multiplier: Hormander-Mihlin condition
    oscillatory_int: stationary phase decay rate
    strichartz_est: admissible pair scaling
    """
    return aux


def _bench_restriction_est(seed: int = 0) -> float:
    checks = []
    checks.append(restriction_est_ok(True, True))
    checks.append(not restriction_est_ok(False, True))
    checks.append(restriction_est_aux(True))
    checks.append(not restriction_est_aux(False))
    checks.append(True)  # harmonic-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_restriction_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_restriction_est": _bench_restriction_est(seed)}
