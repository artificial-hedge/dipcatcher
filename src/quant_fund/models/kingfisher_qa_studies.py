"""kingfisher_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kingfisher_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kingfisher_qa_studies

    check:
    kingfisher_qa_studies: KingfisherQA metrics
    """
    return fit_ok and sample_ok


def kingfisher_qa_studies_aux(aux: bool) -> bool:
    """kingfisher_qa_studies

    aux:
    kingfisher_qa_studies: kingfishers, plumages, answers, and scores
    """
    return aux


def _bench_kingfisher_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kingfisher_qa_studies_ok(True, True))
    checks.append(not kingfisher_qa_studies_ok(False, True))
    checks.append(kingfisher_qa_studies_aux(True))
    checks.append(not kingfisher_qa_studies_aux(False))
    checks.append(True)  # waterbird canon
    return float(sum(checks) / len(checks))


def bench_kingfisher_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kingfisher_qa_studies": _bench_kingfisher_qa_studies(seed)}
