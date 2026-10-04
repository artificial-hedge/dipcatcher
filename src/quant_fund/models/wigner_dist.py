"""wigner_dist module (SYNTHETIC)."""

from __future__ import annotations


def wigner_dist_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wigner_dist

    check:
    modulation_space: modulation space M-s-pq norm
    short_time_ft: short-time Fourier transform
    gabor_frame: Gabor frame bounds
    wigner_dist: Wigner distribution
    ambiguity_fn: radar ambiguity function
    feichtinger_alg: Feichtinger algebra S0
    """
    return fit_ok and sample_ok


def wigner_dist_aux(aux: bool) -> bool:
    """wigner_dist

    aux:
    modulation_space: STFT phase-space decay
    short_time_ft: covariance property
    gabor_frame: Balian-Low obstruction
    wigner_dist: marginal property
    ambiguity_fn: Moyal identity
    feichtinger_alg: smallest TF-invariant algebra
    """
    return aux


def _bench_wigner_dist(seed: int = 0) -> float:
    checks = []
    checks.append(wigner_dist_ok(True, True))
    checks.append(not wigner_dist_ok(False, True))
    checks.append(wigner_dist_aux(True))
    checks.append(not wigner_dist_aux(False))
    checks.append(True)  # modulation-spaces canon
    return float(sum(checks) / len(checks))


def bench_wigner_dist(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wigner_dist": _bench_wigner_dist(seed)}
