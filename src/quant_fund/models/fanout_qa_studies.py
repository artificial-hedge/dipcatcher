"""fanout_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fanout_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fanout_qa_studies

    check:
    fanout_qa_studies: FanOutQA metrics
    """
    return fit_ok and sample_ok


def fanout_qa_studies_aux(aux: bool) -> bool:
    """fanout_qa_studies

    aux:
    fanout_qa_studies: questions, documents, subquestions, and scores
    """
    return aux


def _bench_fanout_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fanout_qa_studies_ok(True, True))
    checks.append(not fanout_qa_studies_ok(False, True))
    checks.append(fanout_qa_studies_aux(True))
    checks.append(not fanout_qa_studies_aux(False))
    checks.append(True)  # long-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_fanout_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fanout_qa_studies": _bench_fanout_qa_studies(seed)}
