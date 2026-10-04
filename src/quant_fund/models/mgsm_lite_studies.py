"""mgsm_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mgsm_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mgsm_lite_studies

    check:
    mgsm_lite_studies: MGSM multilingual math metrics
    """
    return fit_ok and sample_ok


def mgsm_lite_studies_aux(aux: bool) -> bool:
    """mgsm_lite_studies

    aux:
    mgsm_lite_studies: problems, languages, answers, and scores
    """
    return aux


def _bench_mgsm_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mgsm_lite_studies_ok(True, True))
    checks.append(not mgsm_lite_studies_ok(False, True))
    checks.append(mgsm_lite_studies_aux(True))
    checks.append(not mgsm_lite_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_mgsm_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mgsm_lite_studies": _bench_mgsm_lite_studies(seed)}
