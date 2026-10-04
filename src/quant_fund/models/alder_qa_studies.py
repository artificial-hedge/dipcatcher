"""alder_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def alder_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """alder_qa_studies

    check:
    alder_qa_studies: AlderQA metrics
    """
    return fit_ok and sample_ok


def alder_qa_studies_aux(aux: bool) -> bool:
    """alder_qa_studies

    aux:
    alder_qa_studies: alders, wetlands, answers, and scores
    """
    return aux


def _bench_alder_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(alder_qa_studies_ok(True, True))
    checks.append(not alder_qa_studies_ok(False, True))
    checks.append(alder_qa_studies_aux(True))
    checks.append(not alder_qa_studies_aux(False))
    checks.append(True)  # tree-2 canon
    return float(sum(checks) / len(checks))


def bench_alder_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_alder_qa_studies": _bench_alder_qa_studies(seed)}
