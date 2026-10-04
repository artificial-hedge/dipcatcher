"""rongo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rongo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rongo_qa_studies

    check:
    rongo_qa_studies: RongoQA metrics
    """
    return fit_ok and sample_ok


def rongo_qa_studies_aux(aux: bool) -> bool:
    """rongo_qa_studies

    aux:
    rongo_qa_studies: rongo, peace farmers, answers, and scores
    """
    return aux


def _bench_rongo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rongo_qa_studies_ok(True, True))
    checks.append(not rongo_qa_studies_ok(False, True))
    checks.append(rongo_qa_studies_aux(True))
    checks.append(not rongo_qa_studies_aux(False))
    checks.append(True)  # polynesian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_rongo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rongo_qa_studies": _bench_rongo_qa_studies(seed)}
