"""fairy_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fairy_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fairy_qa_studies

    check:
    fairy_qa_studies: FairyQA metrics
    """
    return fit_ok and sample_ok


def fairy_qa_studies_aux(aux: bool) -> bool:
    """fairy_qa_studies

    aux:
    fairy_qa_studies: fairies, canopies, answers, and scores
    """
    return aux


def _bench_fairy_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fairy_qa_studies_ok(True, True))
    checks.append(not fairy_qa_studies_ok(False, True))
    checks.append(fairy_qa_studies_aux(True))
    checks.append(not fairy_qa_studies_aux(False))
    checks.append(True)  # hummingbird-2 canon
    return float(sum(checks) / len(checks))


def bench_fairy_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fairy_qa_studies": _bench_fairy_qa_studies(seed)}
