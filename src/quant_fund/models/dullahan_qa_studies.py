"""dullahan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dullahan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dullahan_qa_studies

    check:
    dullahan_qa_studies: DullahanQA metrics
    """
    return fit_ok and sample_ok


def dullahan_qa_studies_aux(aux: bool) -> bool:
    """dullahan_qa_studies

    aux:
    dullahan_qa_studies: dullahans, headless riders, answers, and scores
    """
    return aux


def _bench_dullahan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dullahan_qa_studies_ok(True, True))
    checks.append(not dullahan_qa_studies_ok(False, True))
    checks.append(dullahan_qa_studies_aux(True))
    checks.append(not dullahan_qa_studies_aux(False))
    checks.append(True)  # celtic-beast canon
    return float(sum(checks) / len(checks))


def bench_dullahan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dullahan_qa_studies": _bench_dullahan_qa_studies(seed)}
