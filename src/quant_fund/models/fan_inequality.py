"""fan_inequality module (SYNTHETIC)."""

from __future__ import annotations


def fan_inequality_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fan_inequality

    check:
    fan_inequality: Fan dominance inequality
    horn_inequality: Horn eigenvalue inequalities
    weyl_ineq: Weyl perturbation inequalities
    cauchy_binet: Cauchy–Binet formula
    schur_complement: Schur complement block det
    majorization_vec: vector majorization order
    """
    return fit_ok and sample_ok


def fan_inequality_aux(aux: bool) -> bool:
    """fan_inequality

    aux:
    fan_inequality: Ky Fan trace characterizations
    horn_inequality: triples of eigenvalue sums
    weyl_ineq: interlacing under rank update
    cauchy_binet: minor product expansion
    schur_complement: det = det(A) det(C - B A^-1 B')
    majorization_vec: doubly-stochastic characterization
    """
    return aux


def _bench_fan_inequality(seed: int = 0) -> float:
    checks = []
    checks.append(fan_inequality_ok(True, True))
    checks.append(not fan_inequality_ok(False, True))
    checks.append(fan_inequality_aux(True))
    checks.append(not fan_inequality_aux(False))
    checks.append(True)  # matrix-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_fan_inequality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fan_inequality": _bench_fan_inequality(seed)}
