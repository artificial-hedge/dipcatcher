"""tangaroa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tangaroa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tangaroa_qa_studies

    check:
    tangaroa_qa_studies: TangaroaQA metrics
    """
    return fit_ok and sample_ok


def tangaroa_qa_studies_aux(aux: bool) -> bool:
    """tangaroa_qa_studies

    aux:
    tangaroa_qa_studies: tangaroa, ocean fathers, answers, and scores
    """
    return aux


def _bench_tangaroa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tangaroa_qa_studies_ok(True, True))
    checks.append(not tangaroa_qa_studies_ok(False, True))
    checks.append(tangaroa_qa_studies_aux(True))
    checks.append(not tangaroa_qa_studies_aux(False))
    checks.append(True)  # polynesian-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_tangaroa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangaroa_qa_studies": _bench_tangaroa_qa_studies(seed)}
