"""qasper_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def qasper_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qasper_qa_studies

    check:
    qasper_qa_studies: QASPER paper-reading metrics
    """
    return fit_ok and sample_ok


def qasper_qa_studies_aux(aux: bool) -> bool:
    """qasper_qa_studies

    aux:
    qasper_qa_studies: papers, questions, answers, and f1
    """
    return aux


def _bench_qasper_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qasper_qa_studies_ok(True, True))
    checks.append(not qasper_qa_studies_ok(False, True))
    checks.append(qasper_qa_studies_aux(True))
    checks.append(not qasper_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-3 canon
    return float(sum(checks) / len(checks))


def bench_qasper_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qasper_qa_studies": _bench_qasper_qa_studies(seed)}
