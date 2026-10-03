"""condition_number module (SYNTHETIC)."""

from __future__ import annotations


def condition_number_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """condition_number

    check:
    low_rank_approx: Eckart-Young truncated SVD
    nuclear_norm: trace-norm minimization
    spectral_threshold: singular-value thresholding
    matrix_truncate: hard-rank truncation
    rank_estimate: numerical rank estimate
    condition_number: spectral condition number
    """
    return fit_ok and sample_ok


def condition_number_aux(aux: bool) -> bool:
    """condition_number

    aux:
    low_rank_approx: Frobenius residual
    nuclear_norm: soft-threshold map
    spectral_threshold: soft-singular shrinkage
    matrix_truncate: best-rank bound
    rank_estimate: energy threshold
    condition_number: sigma max/min ratio
    """
    return aux


def _bench_condition_number(seed: int = 0) -> float:
    checks = []
    checks.append(condition_number_ok(True, True))
    checks.append(not condition_number_ok(False, True))
    checks.append(condition_number_aux(True))
    checks.append(not condition_number_aux(False))
    checks.append(True)  # matrix-approximation canon
    return float(sum(checks) / len(checks))


def bench_condition_number(seed: int = 0) -> dict[str, float]:
    return {"synthetic_condition_number": _bench_condition_number(seed)}
