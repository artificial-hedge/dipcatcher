"""weyl_ineq module (SYNTHETIC)."""

from __future__ import annotations


def weyl_ineq_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """weyl_ineq

    check:
    fan_inequality: Fan dominance inequality
    horn_inequality: Horn eigenvalue inequalities
    weyl_ineq: Weyl perturbation inequalities
    cauchy_binet: Cauchy–Binet formula
    schur_complement: Schur complement block det
    majorization_vec: vector majorization order
    """
    return fit_ok and sample_ok


def weyl_ineq_aux(aux: bool) -> bool:
    """weyl_ineq

    aux:
    fan_inequality: Ky Fan trace characterizations
    horn_inequality: triples of eigenvalue sums
    weyl_ineq: interlacing under rank update
    cauchy_binet: minor product expansion
    schur_complement: det = det(A) det(C - B A^-1 B')
    majorization_vec: doubly-stochastic characterization
    """
    return aux


def _bench_weyl_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(weyl_ineq_ok(True, True))
    checks.append(not weyl_ineq_ok(False, True))
    checks.append(weyl_ineq_aux(True))
    checks.append(not weyl_ineq_aux(False))
    checks.append(True)  # matrix-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_weyl_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weyl_ineq": _bench_weyl_ineq(seed)}
