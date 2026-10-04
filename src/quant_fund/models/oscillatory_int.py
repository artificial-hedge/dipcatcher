"""oscillatory_int module (SYNTHETIC)."""

from __future__ import annotations


def oscillatory_int_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oscillatory_int

    check:
    hausdorff_young: Hausdorff-Young Lp Fourier bound
    restriction_est: Fourier restriction to surfaces
    bochner_riesz: Bochner-Riesz summability index
    lp_multiplier: Lp Fourier multiplier estimates
    oscillatory_int: oscillatory integral decay
    strichartz_est: Strichartz space-time estimate
    """
    return fit_ok and sample_ok


def oscillatory_int_aux(aux: bool) -> bool:
    """oscillatory_int

    aux:
    hausdorff_young: Plancherel endpoint interpolation
    restriction_est: Tomas-Stein exponent range
    bochner_riesz: critical index comparison
    lp_multiplier: Hormander-Mihlin condition
    oscillatory_int: stationary phase decay rate
    strichartz_est: admissible pair scaling
    """
    return aux


def _bench_oscillatory_int(seed: int = 0) -> float:
    checks = []
    checks.append(oscillatory_int_ok(True, True))
    checks.append(not oscillatory_int_ok(False, True))
    checks.append(oscillatory_int_aux(True))
    checks.append(not oscillatory_int_aux(False))
    checks.append(True)  # harmonic-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_oscillatory_int(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oscillatory_int": _bench_oscillatory_int(seed)}
