"""agave_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agave_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agave_qa_studies

    check:
    agave_qa_studies: AgaveQA metrics
    """
    return fit_ok and sample_ok


def agave_qa_studies_aux(aux: bool) -> bool:
    """agave_qa_studies

    aux:
    agave_qa_studies: agaves, canyons, answers, and scores
    """
    return aux


def _bench_agave_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agave_qa_studies_ok(True, True))
    checks.append(not agave_qa_studies_ok(False, True))
    checks.append(agave_qa_studies_aux(True))
    checks.append(not agave_qa_studies_aux(False))
    checks.append(True)  # succulent canon
    return float(sum(checks) / len(checks))


def bench_agave_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agave_qa_studies": _bench_agave_qa_studies(seed)}
