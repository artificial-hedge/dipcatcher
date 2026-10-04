"""phalarope_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def phalarope_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """phalarope_qa_studies

    check:
    phalarope_qa_studies: PhalaropeQA metrics
    """
    return fit_ok and sample_ok


def phalarope_qa_studies_aux(aux: bool) -> bool:
    """phalarope_qa_studies

    aux:
    phalarope_qa_studies: phalaropes, lagoons, answers, and scores
    """
    return aux


def _bench_phalarope_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(phalarope_qa_studies_ok(True, True))
    checks.append(not phalarope_qa_studies_ok(False, True))
    checks.append(phalarope_qa_studies_aux(True))
    checks.append(not phalarope_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_phalarope_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phalarope_qa_studies": _bench_phalarope_qa_studies(seed)}
