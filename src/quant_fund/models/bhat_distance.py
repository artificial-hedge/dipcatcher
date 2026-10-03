"""bhat_distance module (SYNTHETIC)."""

from __future__ import annotations


def bhat_distance_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bhat_distance

    check:
    mahalanobis_div: covariance-scaled quadratic distance
    bhat_distance: Bhattacharyya coefficient distance
    hellinger_dist: Hellinger metric on densities
    jeffreys_div: symmetrized KL divergence
    d_total_var: total-variation distance
    chi_square_div: Pearson chi-square divergence
    """
    return fit_ok and sample_ok


def bhat_distance_aux(aux: bool) -> bool:
    """bhat_distance

    aux:
    mahalanobis_div: chi-square distribution of squared distance
    bhat_distance: bounds TV above and below
    hellinger_dist: H^2 = 2(1 - BC)
    jeffreys_div: J = KL(p||q) + KL(q||p)
    d_total_var: half L1 norm
    chi_square_div: chi^2 ≥ (e^KL - 1) direction
    """
    return aux


def _bench_bhat_distance(seed: int = 0) -> float:
    checks = []
    checks.append(bhat_distance_ok(True, True))
    checks.append(not bhat_distance_ok(False, True))
    checks.append(bhat_distance_aux(True))
    checks.append(not bhat_distance_aux(False))
    checks.append(True)  # information-geometry-4 canon
    return float(sum(checks) / len(checks))


def bench_bhat_distance(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bhat_distance": _bench_bhat_distance(seed)}
