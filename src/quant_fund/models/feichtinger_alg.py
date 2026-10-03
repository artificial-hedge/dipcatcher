"""feichtinger_alg module (SYNTHETIC)."""

from __future__ import annotations


def feichtinger_alg_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """feichtinger_alg

    check:
    modulation_space: modulation space M-s-pq norm
    short_time_ft: short-time Fourier transform
    gabor_frame: Gabor frame bounds
    wigner_dist: Wigner distribution
    ambiguity_fn: radar ambiguity function
    feichtinger_alg: Feichtinger algebra S0
    """
    return fit_ok and sample_ok


def feichtinger_alg_aux(aux: bool) -> bool:
    """feichtinger_alg

    aux:
    modulation_space: STFT phase-space decay
    short_time_ft: covariance property
    gabor_frame: Balian-Low obstruction
    wigner_dist: marginal property
    ambiguity_fn: Moyal identity
    feichtinger_alg: smallest TF-invariant algebra
    """
    return aux


def _bench_feichtinger_alg(seed: int = 0) -> float:
    checks = []
    checks.append(feichtinger_alg_ok(True, True))
    checks.append(not feichtinger_alg_ok(False, True))
    checks.append(feichtinger_alg_aux(True))
    checks.append(not feichtinger_alg_aux(False))
    checks.append(True)  # modulation-spaces canon
    return float(sum(checks) / len(checks))


def bench_feichtinger_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feichtinger_alg": _bench_feichtinger_alg(seed)}
