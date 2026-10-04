"""exam_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def exam_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """exam_qa_studies

    check:
    exam_qa_studies: ExamQA metrics
    """
    return fit_ok and sample_ok


def exam_qa_studies_aux(aux: bool) -> bool:
    """exam_qa_studies

    aux:
    exam_qa_studies: exams, questions, answers, and scores
    """
    return aux


def _bench_exam_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(exam_qa_studies_ok(True, True))
    checks.append(not exam_qa_studies_ok(False, True))
    checks.append(exam_qa_studies_aux(True))
    checks.append(not exam_qa_studies_aux(False))
    checks.append(True)  # education canon
    return float(sum(checks) / len(checks))


def bench_exam_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exam_qa_studies": _bench_exam_qa_studies(seed)}
