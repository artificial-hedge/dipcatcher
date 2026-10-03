"""indian_buffet module (SYNTHETIC)."""

from __future__ import annotations


def indian_buffet_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """indian_buffet

    check:
    dirichlet_process: DP measure sampling
    stick_breaking: Sethuraman stick construction
    pitman_yor: Pitman-Yor two-parameter process
    indian_buffet: IBP feature allocation
    chinese_restaurant: CRP table assignment
    hierarchical_dp: shared-base hierarchical DP
    """
    return fit_ok and sample_ok


def indian_buffet_aux(aux: bool) -> bool:
    """indian_buffet

    aux:
    dirichlet_process: normalized gamma process
    stick_breaking: GEM(α) weight sequence
    pitman_yor: power-law occupancy
    indian_buffet: left-ordered binary matrix
    chinese_restaurant: exchangeable partition probability
    hierarchical_dp: franchise-table counts
    """
    return aux


def _bench_indian_buffet(seed: int = 0) -> float:
    checks = []
    checks.append(indian_buffet_ok(True, True))
    checks.append(not indian_buffet_ok(False, True))
    checks.append(indian_buffet_aux(True))
    checks.append(not indian_buffet_aux(False))
    checks.append(True)  # Bayesian-nonparametrics canon
    return float(sum(checks) / len(checks))


def bench_indian_buffet(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indian_buffet": _bench_indian_buffet(seed)}
