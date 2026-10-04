"""boolq_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boolq_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boolq_qa_studies

    check:
    boolq_qa_studies: BoolQ yes/no metrics
    """
    return fit_ok and sample_ok


def boolq_qa_studies_aux(aux: bool) -> bool:
    """boolq_qa_studies

    aux:
    boolq_qa_studies: passages, questions, labels, and accuracies
    """
    return aux


def _bench_boolq_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boolq_qa_studies_ok(True, True))
    checks.append(not boolq_qa_studies_ok(False, True))
    checks.append(boolq_qa_studies_aux(True))
    checks.append(not boolq_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-3 canon
    return float(sum(checks) / len(checks))


def bench_boolq_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boolq_qa_studies": _bench_boolq_qa_studies(seed)}
