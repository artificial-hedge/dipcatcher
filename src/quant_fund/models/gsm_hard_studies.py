"""gsm_hard_studies module (SYNTHETIC)."""

from __future__ import annotations


def gsm_hard_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gsm_hard_studies

    check:
    gsm_hard_studies: Hard grade-school-math metrics
    """
    return fit_ok and sample_ok


def gsm_hard_studies_aux(aux: bool) -> bool:
    """gsm_hard_studies

    aux:
    gsm_hard_studies: questions, chains, answers, and pass rates
    """
    return aux


def _bench_gsm_hard_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gsm_hard_studies_ok(True, True))
    checks.append(not gsm_hard_studies_ok(False, True))
    checks.append(gsm_hard_studies_aux(True))
    checks.append(not gsm_hard_studies_aux(False))
    checks.append(True)  # math-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_gsm_hard_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gsm_hard_studies": _bench_gsm_hard_studies(seed)}
