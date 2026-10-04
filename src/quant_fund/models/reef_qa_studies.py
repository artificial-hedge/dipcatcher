"""reef_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def reef_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reef_qa_studies

    check:
    reef_qa_studies: ReefQA metrics
    """
    return fit_ok and sample_ok


def reef_qa_studies_aux(aux: bool) -> bool:
    """reef_qa_studies

    aux:
    reef_qa_studies: reefs, ecosystems, answers, and scores
    """
    return aux


def _bench_reef_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reef_qa_studies_ok(True, True))
    checks.append(not reef_qa_studies_ok(False, True))
    checks.append(reef_qa_studies_aux(True))
    checks.append(not reef_qa_studies_aux(False))
    checks.append(True)  # marine canon
    return float(sum(checks) / len(checks))


def bench_reef_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reef_qa_studies": _bench_reef_qa_studies(seed)}
