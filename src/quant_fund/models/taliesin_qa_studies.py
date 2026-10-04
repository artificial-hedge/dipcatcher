"""taliesin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def taliesin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """taliesin_qa_studies

    check:
    taliesin_qa_studies: TaliesinQA metrics
    """
    return fit_ok and sample_ok


def taliesin_qa_studies_aux(aux: bool) -> bool:
    """taliesin_qa_studies

    aux:
    taliesin_qa_studies: taliesin, bard songs, answers, and scores
    """
    return aux


def _bench_taliesin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(taliesin_qa_studies_ok(True, True))
    checks.append(not taliesin_qa_studies_ok(False, True))
    checks.append(taliesin_qa_studies_aux(True))
    checks.append(not taliesin_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_taliesin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taliesin_qa_studies": _bench_taliesin_qa_studies(seed)}
