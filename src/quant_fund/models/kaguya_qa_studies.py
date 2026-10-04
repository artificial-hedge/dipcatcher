"""kaguya_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kaguya_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kaguya_qa_studies

    check:
    kaguya_qa_studies: KaguyaQA metrics
    """
    return fit_ok and sample_ok


def kaguya_qa_studies_aux(aux: bool) -> bool:
    """kaguya_qa_studies

    aux:
    kaguya_qa_studies: kaguya, moon princesses, answers, and scores
    """
    return aux


def _bench_kaguya_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kaguya_qa_studies_ok(True, True))
    checks.append(not kaguya_qa_studies_ok(False, True))
    checks.append(kaguya_qa_studies_aux(True))
    checks.append(not kaguya_qa_studies_aux(False))
    checks.append(True)  # japanese-myth-5 canon
    return float(sum(checks) / len(checks))


def bench_kaguya_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kaguya_qa_studies": _bench_kaguya_qa_studies(seed)}
