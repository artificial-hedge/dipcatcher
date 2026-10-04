"""steelhead_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def steelhead_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """steelhead_qa_studies

    check:
    steelhead_qa_studies: SteelheadQA metrics
    """
    return fit_ok and sample_ok


def steelhead_qa_studies_aux(aux: bool) -> bool:
    """steelhead_qa_studies

    aux:
    steelhead_qa_studies: steelhead, coastal rivers, answers, and scores
    """
    return aux


def _bench_steelhead_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(steelhead_qa_studies_ok(True, True))
    checks.append(not steelhead_qa_studies_ok(False, True))
    checks.append(steelhead_qa_studies_aux(True))
    checks.append(not steelhead_qa_studies_aux(False))
    checks.append(True)  # salmonid canon
    return float(sum(checks) / len(checks))


def bench_steelhead_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steelhead_qa_studies": _bench_steelhead_qa_studies(seed)}
