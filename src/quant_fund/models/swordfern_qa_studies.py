"""swordfern_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def swordfern_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swordfern_qa_studies

    check:
    swordfern_qa_studies: SwordfernQA metrics
    """
    return fit_ok and sample_ok


def swordfern_qa_studies_aux(aux: bool) -> bool:
    """swordfern_qa_studies

    aux:
    swordfern_qa_studies: swordferns, understories, answers, and scores
    """
    return aux


def _bench_swordfern_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swordfern_qa_studies_ok(True, True))
    checks.append(not swordfern_qa_studies_ok(False, True))
    checks.append(swordfern_qa_studies_aux(True))
    checks.append(not swordfern_qa_studies_aux(False))
    checks.append(True)  # fern canon
    return float(sum(checks) / len(checks))


def bench_swordfern_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swordfern_qa_studies": _bench_swordfern_qa_studies(seed)}
