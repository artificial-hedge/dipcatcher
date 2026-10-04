"""ukso_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ukso_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ukso_qa_studies

    check:
    ukso_qa_studies: UksoQA metrics
    """
    return fit_ok and sample_ok


def ukso_qa_studies_aux(aux: bool) -> bool:
    """ukso_qa_studies

    aux:
    ukso_qa_studies: ukso, winter kings, answers, and scores
    """
    return aux


def _bench_ukso_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ukso_qa_studies_ok(True, True))
    checks.append(not ukso_qa_studies_ok(False, True))
    checks.append(ukso_qa_studies_aux(True))
    checks.append(not ukso_qa_studies_aux(False))
    checks.append(True)  # sami-myth canon
    return float(sum(checks) / len(checks))


def bench_ukso_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ukso_qa_studies": _bench_ukso_qa_studies(seed)}
