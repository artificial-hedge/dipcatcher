"""thunderbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def thunderbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thunderbird_qa_studies

    check:
    thunderbird_qa_studies: ThunderbirdQA metrics
    """
    return fit_ok and sample_ok


def thunderbird_qa_studies_aux(aux: bool) -> bool:
    """thunderbird_qa_studies

    aux:
    thunderbird_qa_studies: thunderbirds, storm peaks, answers, and scores
    """
    return aux


def _bench_thunderbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(thunderbird_qa_studies_ok(True, True))
    checks.append(not thunderbird_qa_studies_ok(False, True))
    checks.append(thunderbird_qa_studies_aux(True))
    checks.append(not thunderbird_qa_studies_aux(False))
    checks.append(True)  # cryptid canon
    return float(sum(checks) / len(checks))


def bench_thunderbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thunderbird_qa_studies": _bench_thunderbird_qa_studies(seed)}
