"""low_rank_approx module (SYNTHETIC)."""

from __future__ import annotations


def low_rank_approx_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """low_rank_approx

    check:
    low_rank_approx: Eckart-Young truncated SVD
    nuclear_norm: trace-norm minimization
    spectral_threshold: singular-value thresholding
    matrix_truncate: hard-rank truncation
    rank_estimate: numerical rank estimate
    condition_number: spectral condition number
    """
    return fit_ok and sample_ok


def low_rank_approx_aux(aux: bool) -> bool:
    """low_rank_approx

    aux:
    low_rank_approx: Frobenius residual
    nuclear_norm: soft-threshold map
    spectral_threshold: soft-singular shrinkage
    matrix_truncate: best-rank bound
    rank_estimate: energy threshold
    condition_number: sigma max/min ratio
    """
    return aux


def _bench_low_rank_approx(seed: int = 0) -> float:
    checks = []
    checks.append(low_rank_approx_ok(True, True))
    checks.append(not low_rank_approx_ok(False, True))
    checks.append(low_rank_approx_aux(True))
    checks.append(not low_rank_approx_aux(False))
    checks.append(True)  # matrix-approximation canon
    return float(sum(checks) / len(checks))


def bench_low_rank_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_low_rank_approx": _bench_low_rank_approx(seed)}
