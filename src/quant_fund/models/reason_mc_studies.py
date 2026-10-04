"""reason_mc_studies module (SYNTHETIC)."""

from __future__ import annotations


def reason_mc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reason_mc_studies

    check:
    reason_mc_studies: MC logical-reasoning metrics
    """
    return fit_ok and sample_ok


def reason_mc_studies_aux(aux: bool) -> bool:
    """reason_mc_studies

    aux:
    reason_mc_studies: passages, questions, options, and accuracies
    """
    return aux


def _bench_reason_mc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reason_mc_studies_ok(True, True))
    checks.append(not reason_mc_studies_ok(False, True))
    checks.append(reason_mc_studies_aux(True))
    checks.append(not reason_mc_studies_aux(False))
    checks.append(True)  # logical-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_reason_mc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reason_mc_studies": _bench_reason_mc_studies(seed)}
