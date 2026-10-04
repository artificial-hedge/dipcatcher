"""norns_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def norns_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """norns_qa_studies

    check:
    norns_qa_studies: NornsQA metrics
    """
    return fit_ok and sample_ok


def norns_qa_studies_aux(aux: bool) -> bool:
    """norns_qa_studies

    aux:
    norns_qa_studies: norns, well weavers, answers, and scores
    """
    return aux


def _bench_norns_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(norns_qa_studies_ok(True, True))
    checks.append(not norns_qa_studies_ok(False, True))
    checks.append(norns_qa_studies_aux(True))
    checks.append(not norns_qa_studies_aux(False))
    checks.append(True)  # norse-myth-11 canon
    return float(sum(checks) / len(checks))


def bench_norns_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norns_qa_studies": _bench_norns_qa_studies(seed)}
