"""grayling_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grayling_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grayling_qa_studies

    check:
    grayling_qa_studies: GraylingQA metrics
    """
    return fit_ok and sample_ok


def grayling_qa_studies_aux(aux: bool) -> bool:
    """grayling_qa_studies

    aux:
    grayling_qa_studies: graylings, boreal rivers, answers, and scores
    """
    return aux


def _bench_grayling_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grayling_qa_studies_ok(True, True))
    checks.append(not grayling_qa_studies_ok(False, True))
    checks.append(grayling_qa_studies_aux(True))
    checks.append(not grayling_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_grayling_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grayling_qa_studies": _bench_grayling_qa_studies(seed)}
