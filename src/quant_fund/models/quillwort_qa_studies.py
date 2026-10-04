"""quillwort_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quillwort_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quillwort_qa_studies

    check:
    quillwort_qa_studies: QuillwortQA metrics
    """
    return fit_ok and sample_ok


def quillwort_qa_studies_aux(aux: bool) -> bool:
    """quillwort_qa_studies

    aux:
    quillwort_qa_studies: quillworts, lakebeds, answers, and scores
    """
    return aux


def _bench_quillwort_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quillwort_qa_studies_ok(True, True))
    checks.append(not quillwort_qa_studies_ok(False, True))
    checks.append(quillwort_qa_studies_aux(True))
    checks.append(not quillwort_qa_studies_aux(False))
    checks.append(True)  # moss canon
    return float(sum(checks) / len(checks))


def bench_quillwort_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quillwort_qa_studies": _bench_quillwort_qa_studies(seed)}
