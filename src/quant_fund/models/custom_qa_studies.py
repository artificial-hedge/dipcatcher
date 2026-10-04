"""custom_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def custom_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """custom_qa_studies

    check:
    custom_qa_studies: CustomQA metrics
    """
    return fit_ok and sample_ok


def custom_qa_studies_aux(aux: bool) -> bool:
    """custom_qa_studies

    aux:
    custom_qa_studies: contexts, customs, answers, and scores
    """
    return aux


def _bench_custom_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(custom_qa_studies_ok(True, True))
    checks.append(not custom_qa_studies_ok(False, True))
    checks.append(custom_qa_studies_aux(True))
    checks.append(not custom_qa_studies_aux(False))
    checks.append(True)  # folk-commonsense canon
    return float(sum(checks) / len(checks))


def bench_custom_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_custom_qa_studies": _bench_custom_qa_studies(seed)}
