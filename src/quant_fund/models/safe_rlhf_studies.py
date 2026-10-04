"""safe_rlhf_studies module (SYNTHETIC)."""

from __future__ import annotations


def safe_rlhf_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """safe_rlhf_studies

    check:
    safe_rlhf_studies: Safe-RLHF helpfulness/harmlessness tradeoff metrics
    """
    return fit_ok and sample_ok


def safe_rlhf_studies_aux(aux: bool) -> bool:
    """safe_rlhf_studies

    aux:
    safe_rlhf_studies: prompts, rewards, costs, and safety scores
    """
    return aux


def _bench_safe_rlhf_studies(seed: int = 0) -> float:
    checks = []
    checks.append(safe_rlhf_studies_ok(True, True))
    checks.append(not safe_rlhf_studies_ok(False, True))
    checks.append(safe_rlhf_studies_aux(True))
    checks.append(not safe_rlhf_studies_aux(False))
    checks.append(True)  # safety-alignment-2 canon
    return float(sum(checks) / len(checks))


def bench_safe_rlhf_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_safe_rlhf_studies": _bench_safe_rlhf_studies(seed)}
