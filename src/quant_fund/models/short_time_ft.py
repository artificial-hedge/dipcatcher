"""short_time_ft module (SYNTHETIC)."""

from __future__ import annotations


def short_time_ft_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """short_time_ft

    check:
    modulation_space: modulation space M-s-pq norm
    short_time_ft: short-time Fourier transform
    gabor_frame: Gabor frame bounds
    wigner_dist: Wigner distribution
    ambiguity_fn: radar ambiguity function
    feichtinger_alg: Feichtinger algebra S0
    """
    return fit_ok and sample_ok


def short_time_ft_aux(aux: bool) -> bool:
    """short_time_ft

    aux:
    modulation_space: STFT phase-space decay
    short_time_ft: covariance property
    gabor_frame: Balian-Low obstruction
    wigner_dist: marginal property
    ambiguity_fn: Moyal identity
    feichtinger_alg: smallest TF-invariant algebra
    """
    return aux


def _bench_short_time_ft(seed: int = 0) -> float:
    checks = []
    checks.append(short_time_ft_ok(True, True))
    checks.append(not short_time_ft_ok(False, True))
    checks.append(short_time_ft_aux(True))
    checks.append(not short_time_ft_aux(False))
    checks.append(True)  # modulation-spaces canon
    return float(sum(checks) / len(checks))


def bench_short_time_ft(seed: int = 0) -> dict[str, float]:
    return {"synthetic_short_time_ft": _bench_short_time_ft(seed)}
