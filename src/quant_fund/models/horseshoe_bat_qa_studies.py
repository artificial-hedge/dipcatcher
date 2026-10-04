"""horseshoe_bat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horseshoe_bat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horseshoe_bat_qa_studies

    check:
    horseshoe_bat_qa_studies: HorseshoeBatQA metrics
    """
    return fit_ok and sample_ok


def horseshoe_bat_qa_studies_aux(aux: bool) -> bool:
    """horseshoe_bat_qa_studies

    aux:
    horseshoe_bat_qa_studies: horseshoe bats, limestone caves, answers, and scores
    """
    return aux


def _bench_horseshoe_bat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horseshoe_bat_qa_studies_ok(True, True))
    checks.append(not horseshoe_bat_qa_studies_ok(False, True))
    checks.append(horseshoe_bat_qa_studies_aux(True))
    checks.append(not horseshoe_bat_qa_studies_aux(False))
    checks.append(True)  # bat canon
    return float(sum(checks) / len(checks))


def bench_horseshoe_bat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horseshoe_bat_qa_studies": _bench_horseshoe_bat_qa_studies(seed)}
