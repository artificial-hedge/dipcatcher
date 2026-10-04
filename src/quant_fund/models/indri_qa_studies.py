"""indri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def indri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """indri_qa_studies

    check:
    indri_qa_studies: IndriQA metrics
    """
    return fit_ok and sample_ok


def indri_qa_studies_aux(aux: bool) -> bool:
    """indri_qa_studies

    aux:
    indri_qa_studies: indris, mist forests, answers, and scores
    """
    return aux


def _bench_indri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(indri_qa_studies_ok(True, True))
    checks.append(not indri_qa_studies_ok(False, True))
    checks.append(indri_qa_studies_aux(True))
    checks.append(not indri_qa_studies_aux(False))
    checks.append(True)  # prosimian canon
    return float(sum(checks) / len(checks))


def bench_indri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indri_qa_studies": _bench_indri_qa_studies(seed)}
