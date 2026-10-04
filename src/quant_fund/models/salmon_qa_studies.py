"""salmon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def salmon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """salmon_qa_studies

    check:
    salmon_qa_studies: SalmonQA metrics
    """
    return fit_ok and sample_ok


def salmon_qa_studies_aux(aux: bool) -> bool:
    """salmon_qa_studies

    aux:
    salmon_qa_studies: salmon, runs, answers, and scores
    """
    return aux


def _bench_salmon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(salmon_qa_studies_ok(True, True))
    checks.append(not salmon_qa_studies_ok(False, True))
    checks.append(salmon_qa_studies_aux(True))
    checks.append(not salmon_qa_studies_aux(False))
    checks.append(True)  # fish canon
    return float(sum(checks) / len(checks))


def bench_salmon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_salmon_qa_studies": _bench_salmon_qa_studies(seed)}
