"""goby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def goby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """goby_qa_studies

    check:
    goby_qa_studies: GobyQA metrics
    """
    return fit_ok and sample_ok


def goby_qa_studies_aux(aux: bool) -> bool:
    """goby_qa_studies

    aux:
    goby_qa_studies: gobies, sandy burrows, answers, and scores
    """
    return aux


def _bench_goby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(goby_qa_studies_ok(True, True))
    checks.append(not goby_qa_studies_ok(False, True))
    checks.append(goby_qa_studies_aux(True))
    checks.append(not goby_qa_studies_aux(False))
    checks.append(True)  # reef-fish-2 canon
    return float(sum(checks) / len(checks))


def bench_goby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_goby_qa_studies": _bench_goby_qa_studies(seed)}
