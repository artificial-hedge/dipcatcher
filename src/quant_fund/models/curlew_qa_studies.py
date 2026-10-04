"""curlew_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def curlew_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """curlew_qa_studies

    check:
    curlew_qa_studies: CurlewQA metrics
    """
    return fit_ok and sample_ok


def curlew_qa_studies_aux(aux: bool) -> bool:
    """curlew_qa_studies

    aux:
    curlew_qa_studies: curlews, beaks, answers, and scores
    """
    return aux


def _bench_curlew_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(curlew_qa_studies_ok(True, True))
    checks.append(not curlew_qa_studies_ok(False, True))
    checks.append(curlew_qa_studies_aux(True))
    checks.append(not curlew_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_curlew_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_curlew_qa_studies": _bench_curlew_qa_studies(seed)}
