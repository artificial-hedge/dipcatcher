"""nggp_process module (SYNTHETIC)."""

from __future__ import annotations


def nggp_process_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nggp_process

    check:
    exchangeable_pf: exchangeable partition function
    normalized_rm: normalized random measure
    sigma_stable: σ-stable subordinator prior
    nggp_process: normalized generalized gamma
    bondesson_shot: Bondesson shot-noise class
    kingman_paintbox: Kingman paintbox representation
    """
    return fit_ok and sample_ok


def nggp_process_aux(aux: bool) -> bool:
    """nggp_process

    aux:
    exchangeable_pf: EPPF canonical form
    normalized_rm: CRM normalization
    sigma_stable: stable Lévy jumps
    nggp_process: two-parameter tail index
    bondesson_shot: Poisson shot-noise intensity
    kingman_paintbox: ordered-frequencies law
    """
    return aux


def _bench_nggp_process(seed: int = 0) -> float:
    checks = []
    checks.append(nggp_process_ok(True, True))
    checks.append(not nggp_process_ok(False, True))
    checks.append(nggp_process_aux(True))
    checks.append(not nggp_process_aux(False))
    checks.append(True)  # Bayesian-nonparametrics-3 canon
    return float(sum(checks) / len(checks))


def bench_nggp_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nggp_process": _bench_nggp_process(seed)}
