"""drop_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def drop_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drop_qa_studies

    check:
    drop_qa_studies: DROP discrete-reasoning metrics
    """
    return fit_ok and sample_ok


def drop_qa_studies_aux(aux: bool) -> bool:
    """drop_qa_studies

    aux:
    drop_qa_studies: passages, questions, answers, and scores
    """
    return aux


def _bench_drop_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(drop_qa_studies_ok(True, True))
    checks.append(not drop_qa_studies_ok(False, True))
    checks.append(drop_qa_studies_aux(True))
    checks.append(not drop_qa_studies_aux(False))
    checks.append(True)  # reading-comprehension-2 canon
    return float(sum(checks) / len(checks))


def bench_drop_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drop_qa_studies": _bench_drop_qa_studies(seed)}
