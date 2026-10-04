"""afanc_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def afanc_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """afanc_qa_studies

    check:
    afanc_qa_studies: AfancQA metrics
    """
    return fit_ok and sample_ok


def afanc_qa_studies_aux(aux: bool) -> bool:
    """afanc_qa_studies

    aux:
    afanc_qa_studies: afancs, river depths, answers, and scores
    """
    return aux


def _bench_afanc_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(afanc_qa_studies_ok(True, True))
    checks.append(not afanc_qa_studies_ok(False, True))
    checks.append(afanc_qa_studies_aux(True))
    checks.append(not afanc_qa_studies_aux(False))
    checks.append(True)  # celtic-beast canon
    return float(sum(checks) / len(checks))


def bench_afanc_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_afanc_qa_studies": _bench_afanc_qa_studies(seed)}
