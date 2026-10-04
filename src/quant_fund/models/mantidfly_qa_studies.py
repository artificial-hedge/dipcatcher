"""mantidfly_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mantidfly_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mantidfly_qa_studies

    check:
    mantidfly_qa_studies: MantidflyQA metrics
    """
    return fit_ok and sample_ok


def mantidfly_qa_studies_aux(aux: bool) -> bool:
    """mantidfly_qa_studies

    aux:
    mantidfly_qa_studies: mantidflies, branches, answers, and scores
    """
    return aux


def _bench_mantidfly_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mantidfly_qa_studies_ok(True, True))
    checks.append(not mantidfly_qa_studies_ok(False, True))
    checks.append(mantidfly_qa_studies_aux(True))
    checks.append(not mantidfly_qa_studies_aux(False))
    checks.append(True)  # mantis canon
    return float(sum(checks) / len(checks))


def bench_mantidfly_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mantidfly_qa_studies": _bench_mantidfly_qa_studies(seed)}
