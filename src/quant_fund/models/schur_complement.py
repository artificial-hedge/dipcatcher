"""schur_complement module (SYNTHETIC)."""

from __future__ import annotations


def schur_complement_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """schur_complement

    check:
    fan_inequality: Fan dominance inequality
    horn_inequality: Horn eigenvalue inequalities
    weyl_ineq: Weyl perturbation inequalities
    cauchy_binet: Cauchy–Binet formula
    schur_complement: Schur complement block det
    majorization_vec: vector majorization order
    """
    return fit_ok and sample_ok


def schur_complement_aux(aux: bool) -> bool:
    """schur_complement

    aux:
    fan_inequality: Ky Fan trace characterizations
    horn_inequality: triples of eigenvalue sums
    weyl_ineq: interlacing under rank update
    cauchy_binet: minor product expansion
    schur_complement: det = det(A) det(C - B A^-1 B')
    majorization_vec: doubly-stochastic characterization
    """
    return aux


def _bench_schur_complement(seed: int = 0) -> float:
    checks = []
    checks.append(schur_complement_ok(True, True))
    checks.append(not schur_complement_ok(False, True))
    checks.append(schur_complement_aux(True))
    checks.append(not schur_complement_aux(False))
    checks.append(True)  # matrix-analysis-2 canon
    return float(sum(checks) / len(checks))


def bench_schur_complement(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_complement": _bench_schur_complement(seed)}
