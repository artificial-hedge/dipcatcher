"""lancebill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lancebill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lancebill_qa_studies

    check:
    lancebill_qa_studies: LancebillQA metrics
    """
    return fit_ok and sample_ok


def lancebill_qa_studies_aux(aux: bool) -> bool:
    """lancebill_qa_studies

    aux:
    lancebill_qa_studies: lancebills, highlands, answers, and scores
    """
    return aux


def _bench_lancebill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lancebill_qa_studies_ok(True, True))
    checks.append(not lancebill_qa_studies_ok(False, True))
    checks.append(lancebill_qa_studies_aux(True))
    checks.append(not lancebill_qa_studies_aux(False))
    checks.append(True)  # hummingbird-2 canon
    return float(sum(checks) / len(checks))


def bench_lancebill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lancebill_qa_studies": _bench_lancebill_qa_studies(seed)}
