"""weasel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def weasel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """weasel_qa_studies

    check:
    weasel_qa_studies: WeaselQA metrics
    """
    return fit_ok and sample_ok


def weasel_qa_studies_aux(aux: bool) -> bool:
    """weasel_qa_studies

    aux:
    weasel_qa_studies: weasels, fields, answers, and scores
    """
    return aux


def _bench_weasel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(weasel_qa_studies_ok(True, True))
    checks.append(not weasel_qa_studies_ok(False, True))
    checks.append(weasel_qa_studies_aux(True))
    checks.append(not weasel_qa_studies_aux(False))
    checks.append(True)  # mustelid-2 canon
    return float(sum(checks) / len(checks))


def bench_weasel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weasel_qa_studies": _bench_weasel_qa_studies(seed)}
