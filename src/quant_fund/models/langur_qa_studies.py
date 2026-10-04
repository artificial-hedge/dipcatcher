"""langur_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def langur_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """langur_qa_studies

    check:
    langur_qa_studies: LangurQA metrics
    """
    return fit_ok and sample_ok


def langur_qa_studies_aux(aux: bool) -> bool:
    """langur_qa_studies

    aux:
    langur_qa_studies: langurs, temple groves, answers, and scores
    """
    return aux


def _bench_langur_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(langur_qa_studies_ok(True, True))
    checks.append(not langur_qa_studies_ok(False, True))
    checks.append(langur_qa_studies_aux(True))
    checks.append(not langur_qa_studies_aux(False))
    checks.append(True)  # primate canon
    return float(sum(checks) / len(checks))


def bench_langur_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_langur_qa_studies": _bench_langur_qa_studies(seed)}
