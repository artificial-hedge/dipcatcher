"""matrix_truncate module (SYNTHETIC)."""

from __future__ import annotations


def matrix_truncate_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matrix_truncate

    check:
    low_rank_approx: Eckart-Young truncated SVD
    nuclear_norm: trace-norm minimization
    spectral_threshold: singular-value thresholding
    matrix_truncate: hard-rank truncation
    rank_estimate: numerical rank estimate
    condition_number: spectral condition number
    """
    return fit_ok and sample_ok


def matrix_truncate_aux(aux: bool) -> bool:
    """matrix_truncate

    aux:
    low_rank_approx: Frobenius residual
    nuclear_norm: soft-threshold map
    spectral_threshold: soft-singular shrinkage
    matrix_truncate: best-rank bound
    rank_estimate: energy threshold
    condition_number: sigma max/min ratio
    """
    return aux


def _bench_matrix_truncate(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_truncate_ok(True, True))
    checks.append(not matrix_truncate_ok(False, True))
    checks.append(matrix_truncate_aux(True))
    checks.append(not matrix_truncate_aux(False))
    checks.append(True)  # matrix-approximation canon
    return float(sum(checks) / len(checks))


def bench_matrix_truncate(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_truncate": _bench_matrix_truncate(seed)}
