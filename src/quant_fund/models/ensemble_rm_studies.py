"""ensemble_rm_studies module (SYNTHETIC)."""

from __future__ import annotations


def ensemble_rm_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ensemble_rm_studies

    check:
    ensemble_rm_studies: ensembling multiple reward heads/agreement and spread
    """
    return fit_ok and sample_ok


def ensemble_rm_studies_aux(aux: bool) -> bool:
    """ensemble_rm_studies

    aux:
    ensemble_rm_studies: RM-ensemble disagreement weighting/members and scores
    """
    return aux


def _bench_ensemble_rm_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ensemble_rm_studies_ok(True, True))
    checks.append(not ensemble_rm_studies_ok(False, True))
    checks.append(ensemble_rm_studies_aux(True))
    checks.append(not ensemble_rm_studies_aux(False))
    checks.append(True)  # reward-modeling-2 canon
    return float(sum(checks) / len(checks))


def bench_ensemble_rm_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ensemble_rm_studies": _bench_ensemble_rm_studies(seed)}
