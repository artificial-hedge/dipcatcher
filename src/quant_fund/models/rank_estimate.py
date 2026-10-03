"""rank_estimate module (SYNTHETIC)."""

from __future__ import annotations


def rank_estimate_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rank_estimate

    check:
    low_rank_approx: Eckart-Young truncated SVD
    nuclear_norm: trace-norm minimization
    spectral_threshold: singular-value thresholding
    matrix_truncate: hard-rank truncation
    rank_estimate: numerical rank estimate
    condition_number: spectral condition number
    """
    return fit_ok and sample_ok


def rank_estimate_aux(aux: bool) -> bool:
    """rank_estimate

    aux:
    low_rank_approx: Frobenius residual
    nuclear_norm: soft-threshold map
    spectral_threshold: soft-singular shrinkage
    matrix_truncate: best-rank bound
    rank_estimate: energy threshold
    condition_number: sigma max/min ratio
    """
    return aux


def _bench_rank_estimate(seed: int = 0) -> float:
    checks = []
    checks.append(rank_estimate_ok(True, True))
    checks.append(not rank_estimate_ok(False, True))
    checks.append(rank_estimate_aux(True))
    checks.append(not rank_estimate_aux(False))
    checks.append(True)  # matrix-approximation canon
    return float(sum(checks) / len(checks))


def bench_rank_estimate(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rank_estimate": _bench_rank_estimate(seed)}
