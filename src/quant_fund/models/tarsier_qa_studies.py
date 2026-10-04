"""tarsier_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tarsier_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tarsier_qa_studies

    check:
    tarsier_qa_studies: TarsierQA metrics
    """
    return fit_ok and sample_ok


def tarsier_qa_studies_aux(aux: bool) -> bool:
    """tarsier_qa_studies

    aux:
    tarsier_qa_studies: tarsiers, vertical trunks, answers, and scores
    """
    return aux


def _bench_tarsier_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tarsier_qa_studies_ok(True, True))
    checks.append(not tarsier_qa_studies_ok(False, True))
    checks.append(tarsier_qa_studies_aux(True))
    checks.append(not tarsier_qa_studies_aux(False))
    checks.append(True)  # prosimian canon
    return float(sum(checks) / len(checks))


def bench_tarsier_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarsier_qa_studies": _bench_tarsier_qa_studies(seed)}
