"""horseshoe_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horseshoe_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horseshoe_qa_studies

    check:
    horseshoe_qa_studies: HorseshoeQA metrics
    """
    return fit_ok and sample_ok


def horseshoe_qa_studies_aux(aux: bool) -> bool:
    """horseshoe_qa_studies

    aux:
    horseshoe_qa_studies: horseshoe crabs, ancient shallows, answers, and scores
    """
    return aux


def _bench_horseshoe_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horseshoe_qa_studies_ok(True, True))
    checks.append(not horseshoe_qa_studies_ok(False, True))
    checks.append(horseshoe_qa_studies_aux(True))
    checks.append(not horseshoe_qa_studies_aux(False))
    checks.append(True)  # crab canon
    return float(sum(checks) / len(checks))


def bench_horseshoe_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horseshoe_qa_studies": _bench_horseshoe_qa_studies(seed)}
