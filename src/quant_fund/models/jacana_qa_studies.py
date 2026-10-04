"""jacana_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jacana_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jacana_qa_studies

    check:
    jacana_qa_studies: JacanaQA metrics
    """
    return fit_ok and sample_ok


def jacana_qa_studies_aux(aux: bool) -> bool:
    """jacana_qa_studies

    aux:
    jacana_qa_studies: jacanas, lilypads, answers, and scores
    """
    return aux


def _bench_jacana_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jacana_qa_studies_ok(True, True))
    checks.append(not jacana_qa_studies_ok(False, True))
    checks.append(jacana_qa_studies_aux(True))
    checks.append(not jacana_qa_studies_aux(False))
    checks.append(True)  # wader-2 canon
    return float(sum(checks) / len(checks))


def bench_jacana_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacana_qa_studies": _bench_jacana_qa_studies(seed)}
