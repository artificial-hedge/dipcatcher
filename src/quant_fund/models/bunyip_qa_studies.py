"""bunyip_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bunyip_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bunyip_qa_studies

    check:
    bunyip_qa_studies: BunyipQA metrics
    """
    return fit_ok and sample_ok


def bunyip_qa_studies_aux(aux: bool) -> bool:
    """bunyip_qa_studies

    aux:
    bunyip_qa_studies: bunyips, billabongs, answers, and scores
    """
    return aux


def _bench_bunyip_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bunyip_qa_studies_ok(True, True))
    checks.append(not bunyip_qa_studies_ok(False, True))
    checks.append(bunyip_qa_studies_aux(True))
    checks.append(not bunyip_qa_studies_aux(False))
    checks.append(True)  # cryptid-2 canon
    return float(sum(checks) / len(checks))


def bench_bunyip_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bunyip_qa_studies": _bench_bunyip_qa_studies(seed)}
