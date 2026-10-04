"""koel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def koel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """koel_qa_studies

    check:
    koel_qa_studies: KoelQA metrics
    """
    return fit_ok and sample_ok


def koel_qa_studies_aux(aux: bool) -> bool:
    """koel_qa_studies

    aux:
    koel_qa_studies: koels, mangoes, answers, and scores
    """
    return aux


def _bench_koel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(koel_qa_studies_ok(True, True))
    checks.append(not koel_qa_studies_ok(False, True))
    checks.append(koel_qa_studies_aux(True))
    checks.append(not koel_qa_studies_aux(False))
    checks.append(True)  # nightbird canon
    return float(sum(checks) / len(checks))


def bench_koel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koel_qa_studies": _bench_koel_qa_studies(seed)}
