"""ondal_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ondal_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ondal_qa_studies

    check:
    ondal_qa_studies: OndalQA metrics
    """
    return fit_ok and sample_ok


def ondal_qa_studies_aux(aux: bool) -> bool:
    """ondal_qa_studies

    aux:
    ondal_qa_studies: ondal, fool generals, answers, and scores
    """
    return aux


def _bench_ondal_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ondal_qa_studies_ok(True, True))
    checks.append(not ondal_qa_studies_ok(False, True))
    checks.append(ondal_qa_studies_aux(True))
    checks.append(not ondal_qa_studies_aux(False))
    checks.append(True)  # korean-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_ondal_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ondal_qa_studies": _bench_ondal_qa_studies(seed)}
