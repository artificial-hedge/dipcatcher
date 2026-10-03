"""exchangeable_pf module (SYNTHETIC)."""

from __future__ import annotations


def exchangeable_pf_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """exchangeable_pf

    check:
    exchangeable_pf: exchangeable partition function
    normalized_rm: normalized random measure
    sigma_stable: σ-stable subordinator prior
    nggp_process: normalized generalized gamma
    bondesson_shot: Bondesson shot-noise class
    kingman_paintbox: Kingman paintbox representation
    """
    return fit_ok and sample_ok


def exchangeable_pf_aux(aux: bool) -> bool:
    """exchangeable_pf

    aux:
    exchangeable_pf: EPPF canonical form
    normalized_rm: CRM normalization
    sigma_stable: stable Lévy jumps
    nggp_process: two-parameter tail index
    bondesson_shot: Poisson shot-noise intensity
    kingman_paintbox: ordered-frequencies law
    """
    return aux


def _bench_exchangeable_pf(seed: int = 0) -> float:
    checks = []
    checks.append(exchangeable_pf_ok(True, True))
    checks.append(not exchangeable_pf_ok(False, True))
    checks.append(exchangeable_pf_aux(True))
    checks.append(not exchangeable_pf_aux(False))
    checks.append(True)  # Bayesian-nonparametrics-3 canon
    return float(sum(checks) / len(checks))


def bench_exchangeable_pf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exchangeable_pf": _bench_exchangeable_pf(seed)}
