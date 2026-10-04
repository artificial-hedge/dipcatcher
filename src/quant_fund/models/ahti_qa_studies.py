"""ahti_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ahti_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ahti_qa_studies

    check:
    ahti_qa_studies: AhtiQA metrics
    """
    return fit_ok and sample_ok


def ahti_qa_studies_aux(aux: bool) -> bool:
    """ahti_qa_studies

    aux:
    ahti_qa_studies: ahti, water spirits, answers, and scores
    """
    return aux


def _bench_ahti_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ahti_qa_studies_ok(True, True))
    checks.append(not ahti_qa_studies_ok(False, True))
    checks.append(ahti_qa_studies_aux(True))
    checks.append(not ahti_qa_studies_aux(False))
    checks.append(True)  # finno-ugric-myth canon
    return float(sum(checks) / len(checks))


def bench_ahti_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ahti_qa_studies": _bench_ahti_qa_studies(seed)}
